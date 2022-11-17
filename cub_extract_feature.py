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
from torchvision import datasets, models, transforms
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import torchvision
inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
inat_resnet.load_state_dict(my_model_state_dict, strict=True)
# Freeze backbone (for training only)
for param in list(inat_resnet.parameters())[:-2]:
    param.requires_grad = False
# to CUDA
inat_resnet.cuda()

MODEL1 = inat_resnet

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)
fc = MODEL1.fc
fc = fc.cuda()

Dataset = Dataset()
RunningParams = RunningParams()

RETRIEVE_TOP1_NEAREST = True


in_features = 2048
print("Building FAISS index...")
faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/RN50_dataset_CUB/train/combined', transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if RETRIEVE_TOP1_NEAREST is True:
    INDEX_FILE = 'faiss/faiss_CUB200_class_idx_dict_simclr.npy'
    if os.path.exists(INDEX_FILE):
        print("FAISS class index exists!")
        faiss_nns_class_dict = np.load(INDEX_FILE, allow_pickle="False", ).item()
        targets = faiss_data_loader.dataset.targets
        faiss_data_loader_ids_dict = dict()
        faiss_loader_dict = dict()
        for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
            faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if targets[x] == class_id] # check this value
            class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
            class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
            faiss_loader_dict[class_id] = class_id_loader
    else:
        print("FAISS class index NOT exists!")
        targets = faiss_data_loader.dataset.targets
        faiss_data_loader_ids_dict = dict()
        faiss_nns_class_dict = dict()
        faiss_loader_dict = dict()
        for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
            faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if targets[x] == class_id]
            class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
            class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
            stack_embeddings = []
            for batch_idx, (data, label) in enumerate(class_id_loader):
                input_data = data.detach()
                embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
                embeddings = torch.flatten(embeddings, start_dim=1)

                stack_embeddings.append(embeddings.cpu().detach().numpy())
            stack_embeddings = np.concatenate(stack_embeddings, axis=0)
            descriptors = np.vstack(stack_embeddings)

            cpu_index = faiss.IndexFlatL2(in_features)
            faiss_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
                cpu_index
            )
            faiss_gpu_index = cpu_index

            faiss_gpu_index.add(descriptors)
            faiss_nns_class_dict[class_id] = faiss_gpu_index
            faiss_loader_dict[class_id] = class_id_loader
        np.save(INDEX_FILE, faiss_nns_class_dict)
else:
    INDEX_FILE = 'faiss/faiss_CUB200__topk_class_idx_dict_simclr.index'
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
        input_data = []

        for batch_idx, (data, label) in enumerate(tqdm(faiss_data_loader)):
            input_data = data.detach()
            embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
            embeddings = torch.flatten(embeddings, start_dim=1)

            stack_embeddings.append(embeddings.cpu().detach().numpy())
            stack_labels.append(label.cpu().detach().numpy())

        stack_embeddings = np.concatenate(stack_embeddings, axis=0)
        stack_labels = np.concatenate(stack_labels, axis=0)

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
        faiss.write_index(faiss_cpu_index, INDEX_FILE)


data_dir = '/home/giang/Downloads/RN50_dataset_CUB/test/combined'

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(data_dir, Dataset.data_transforms['train'])

MODEL1 = nn.DataParallel(MODEL1)
train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=128,
    shuffle=True,  # Don't turn shuffle to False --> model works wrongly
    num_workers=16,
    pin_memory=True,
)


faiss_nn_dict = dict()
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    if RETRIEVE_TOP1_NEAREST is True:
        out = MODEL1(data.cuda())
        embeddings = embeddings.cpu().detach().numpy()
        model1_p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(model1_p, 1, dim=1)
        for sample_idx in range(data.shape[0]):
            base_name = os.path.basename(paths[sample_idx])
            predicted_idx = index[sample_idx].item()

            # Dataloader and knowledge base
            loader = faiss_loader_dict[predicted_idx]
            faiss_index = faiss_nns_class_dict[predicted_idx]
            ###############################
            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), 6)
            nn_list = list()
            for id in range(indices.shape[1]):
                id = loader.dataset.indices[indices[0, id]]
                nn_list.append(loader.dataset.dataset.imgs[id][0])
            faiss_nn_dict[base_name] = nn_list
    else:
        embeddings = embeddings.cpu().detach().numpy()
        _, indices = faiss_gpu_index.search(embeddings, 6)  # Retrieve 6 NNs bcz we need to exclude the first one

        for sample_idx in range(data.shape[0]):
            base_name = os.path.basename(paths[sample_idx])
            nn_list = list()
            ids = indices[sample_idx]
            for id in ids:
                nn_list.append(faiss_data_loader.dataset.imgs[id][0])
            faiss_nn_dict[base_name] = nn_list

if RETRIEVE_TOP1_NEAREST:
    np.save('faiss/faiss_CUB_val_top1.npy', faiss_nn_dict)
else:
    np.save('faiss/blah.npy', faiss_nn_dict)
