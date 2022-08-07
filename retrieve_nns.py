import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb
import random
import pdb
import faiss


from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from models import MyCustomResnet18, AdvisingNetwork
from params import RunningParams
from datasets import Dataset
from helpers import HelperFunctions
from explainers import ModelExplainer

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, Convert, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.transforms import *
import torchvision as tv

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

RunningParams = RunningParams()
Dataset = Dataset()
Explainer = ModelExplainer()

model1_name = 'resnet18'
MODEL1 = models.resnet18(pretrained=True).eval().cuda()

data_dir = '/home/giang/Downloads/advising_net_training/'
virtual_train_dataset = '{}/train'.format(data_dir)

train_dataset = '/home/giang/Downloads/datasets/random_train_dataset_10k'
val_dataset = '/home/giang/Downloads/datasets/imagenet5k-1k'
# TODO: change to imagenet-val

if not HelperFunctions.is_running(os.path.basename(__file__)):
    print('Creating symlink datasets...')
    if os.path.islink(virtual_train_dataset) is True:
        os.unlink(virtual_train_dataset)
    os.symlink(train_dataset, virtual_train_dataset)

    virtual_val_dataset = '{}/val'.format(data_dir)
    if os.path.islink(virtual_val_dataset) is True:
        os.unlink(virtual_val_dataset)
    os.symlink(val_dataset, virtual_val_dataset)
else:
    print('Script is running! No creating symlink datasets!')

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      Dataset.data_transforms[x])
              for x in ['train', 'val']}

train_loader = torch.utils.data.DataLoader(
                image_datasets['train'],
                batch_size=RunningParams.batch_size,
                shuffle=True,  # turn shuffle to True
                num_workers=8,
                pin_memory=True,
                )

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
pdist = nn.PairwiseDistance(p=2)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

# TODO: should I also implement this for FFCV?
if RunningParams.XAI_method == RunningParams.NNs:
    in_features = MODEL1.fc.in_features
    print("Building FAISS index...")
    # TODO: change search space to train
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/datasets/random_train_dataset', transform=Dataset.data_transforms['train'])
    faiss_data_loader = torch.utils.data.DataLoader(
        faiss_dataset,
        batch_size=RunningParams.batch_size,
        shuffle=False,  # turn shuffle to True
        num_workers=0,  # Set to 0 as suggested by
        # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
        pin_memory=True,
    )

    INDEX_FILE = 'faiss/faiss.index'
    if os.path.exists(INDEX_FILE):
        print("FAISS index exists!")
        faiss_cpu_index = faiss.read_index(INDEX_FILE)
        faiss_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            faiss_cpu_index
        )
    else:
        print("FAISS index NOT exists! Creating FAISS index then save to disk...")
        stack_embeddings = []
        stack_labels = []
        gallery_paths = []
        for batch_idx, (data, label) in enumerate(tqdm(faiss_data_loader)):
            embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
            embeddings = torch.flatten(embeddings, start_dim=1)
            # print(embeddings.shape)
            # TODO: add data tensors here to return data using index
            # TODO: search in range only
            # TODO: Move everything to GPU
            stack_embeddings.append(embeddings.cpu().detach().numpy())
            stack_labels.append(label.cpu().detach().numpy())
            # gallery_paths.extend(path)

        stack_embeddings = np.concatenate(stack_embeddings, axis=0)
        stack_labels = np.concatenate(stack_labels, axis=0)
        # stack_embeddings = stack_embeddings.cpu().detach().numpy()

        descriptors = np.vstack(stack_embeddings)
        cpu_index = faiss.IndexFlatL2(in_features)
        faiss_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )
        print(faiss_gpu_index.ntotal)

        faiss_gpu_index.add(descriptors)

        faiss_cpu_index = faiss.index_gpu_to_cpu(  # build the index
            faiss_gpu_index
        )
        faiss.write_index(faiss_cpu_index, 'faiss/faiss.index')


for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    embeddings = embeddings.cpu().detach().numpy()
    _, indices = faiss_gpu_index.search(embeddings, 4)  # Retrieve 4 NNs bcz we need to exclude the first one
    # I need to shuffle my training data



