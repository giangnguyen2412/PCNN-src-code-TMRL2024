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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchvision
MODEL1 = torchvision.models.resnet34(pretrained=True).cuda()
MODEL1.eval()
feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor = nn.DataParallel(feature_extractor)
fc = MODEL1.fc

Dataset = Dataset()
RunningParams = RunningParams()

RETRIEVE_TOP1_NEAREST = True

in_features = MODEL1.fc.in_features

imagenet_dataset = ImageFolderWithPaths('/home/giang/Downloads/datasets/imagenet1k-val',
                                         Dataset.data_transforms['train'])

if RunningParams.DOGS_TRAINING == True:
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/datasets/Dogs_train', transform=Dataset.data_transforms['train'])
else:
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/train', transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if RETRIEVE_TOP1_NEAREST is True:
    INDEX_FILE = 'faiss/faiss_Dogs_class_idx_dict_RN34.npy'
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
        print("Building FAISS index........................................")
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
            # faiss_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            #     cpu_index
            # )
            faiss_gpu_index = cpu_index

            faiss_gpu_index.add(descriptors)
            faiss_nns_class_dict[class_id] = faiss_gpu_index
            faiss_loader_dict[class_id] = class_id_loader
        np.save(INDEX_FILE, faiss_nns_class_dict)

#####################################################################################

# Use RN34 to determine whether an image or correct or wrong
resnet34 = models.resnet34(pretrained=True).eval().cuda()
resnet34.eval()
resnet34 = nn.DataParallel(resnet34)

if RunningParams.DOGS_TRAINING is True:
    def load_imagenet_dog_label():
        dog_id_list = list()
        input_f = open("/home/giang/Downloads/SDogs_dataset/dog_type.txt")
        for line in input_f:
            dog_id = (line.split('-')[0])
            dog_id_list.append(dog_id)
        return dog_id_list

    dogs_id = load_imagenet_dog_label()


set = 'test'
data_dir = '/home/giang/Downloads/datasets/'

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                      Dataset.data_transforms['train'])
              for x in ['Dogs_{}'.format(set)]}

train_loader = torch.utils.data.DataLoader(
    image_datasets['Dogs_{}'.format(set)],
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,
    pin_memory=True,
)

dataset_sizes = {x: len(image_datasets[x]) for x in ['Dogs_{}'.format(set)]}

###########################################################################

faiss_nn_dict = dict()
nondog = 0
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    if RETRIEVE_TOP1_NEAREST is True:
        out = resnet34(data.cuda())
        embeddings = embeddings.cpu().detach().numpy()
        model1_p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(model1_p, 1, dim=1)
        for sample_idx in range(data.shape[0]):
            base_name = os.path.basename(paths[sample_idx])
            predicted_idx = index[sample_idx].item()

            if RunningParams.DOGS_TRAINING is True:
                # key = list(imagenet_dataset.class_to_idx.keys())[
                #     list(imagenet_dataset.class_to_idx.values()).index(predicted_idx)]
                # predicted_idx = train_loader.dataset.class_to_idx[key]
                dog_wnid = imagenet_dataset.classes[predicted_idx]
                if dog_wnid not in dogs_id:
                    dog_wnid = random.choice([wnid for wnid in dogs_id if wnid != dog_wnid])
                    nondog += 1

                # dog_idx = train_loader.dataset.class_to_idx[dog_wnid]
                dog_idx = faiss_dataset.class_to_idx[dog_wnid]
                predicted_idx = dog_idx

                gt_idx = label[sample_idx].item()

            # Dataloader and knowledge base
            loader = faiss_loader_dict[predicted_idx]
            faiss_index = faiss_nns_class_dict[predicted_idx]
            ###############################
            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), 11)
            nn_list = list()
            for id in range(indices.shape[1]):
                id = loader.dataset.indices[indices[0, id]]
                nn_list.append(loader.dataset.dataset.imgs[id][0])
            faiss_nn_dict[base_name] = nn_list
    else:
        embeddings = embeddings.cpu().detach().numpy()
        _, indices = faiss_gpu_index.search(embeddings, 11)  # Retrieve 6 NNs bcz we need to exclude the first one

        for sample_idx in range(data.shape[0]):
            base_name = os.path.basename(paths[sample_idx])
            nn_list = list()
            ids = indices[sample_idx]
            for id in ids:
                nn_list.append(faiss_data_loader.dataset.imgs[id][0])
            faiss_nn_dict[base_name] = nn_list

print("Non-dogs: {} files".format(nondog))
if RETRIEVE_TOP1_NEAREST:
    np.save('faiss/dogs/faiss_SDogs_{}_RN34_top1.npy'.format(set), faiss_nn_dict)
else:
    np.save('faiss/dogs/faiss_SDogs_{}_RN34_topk.npy'.format(set), faiss_nn_dict)
