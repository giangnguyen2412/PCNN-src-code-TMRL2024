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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


Dataset = Dataset()
RunningParams = RunningParams()


from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
resnet.load_state_dict(my_model_state_dict, strict=True)
# Freeze backbone (for training only)
for param in list(resnet.parameters())[:-2]:
    param.requires_grad = False
# to CUDA
inat_resnet = resnet.cuda()
MODEL1 = inat_resnet
MODEL1.eval()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

in_features = 2048
print("Building FAISS index...! Training set is the knowledge base.")

faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/nabirds_split_small/train',
                                     transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

INDEX_FILE = 'faiss/cub/NA-Birds-small-zero-shot.npy'
print(INDEX_FILE)

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
    print("FAISS class index NOT exists! Creating class index.........")
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

        faiss_gpu_index = cpu_index

        faiss_gpu_index.add(descriptors)
        faiss_nns_class_dict[class_id] = faiss_gpu_index
        faiss_loader_dict[class_id] = class_id_loader
    np.save(INDEX_FILE, faiss_nns_class_dict)

set = 'test'
data_dir = '/home/giang/Downloads/nabirds_split_small/{}'.format(set)

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(data_dir, Dataset.data_transforms['train'])
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
    embeddings = embeddings.cpu().detach().numpy()
    ################################################################

    for sample_idx in range(data.shape[0]):
        base_name = os.path.basename(paths[sample_idx])

        for key, value in faiss_loader_dict.items():
            # Dataloader and knowledge base upon the predicted class
            loader = faiss_loader_dict[key]
            faiss_index = faiss_nns_class_dict[key]
            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), 6)
            nn_list = list()
            for id in range(indices.shape[1]):
                id = loader.dataset.indices[indices[0, id]]
                nn_list.append(loader.dataset.dataset.imgs[id][0])
            if base_name not in faiss_nn_dict:
                faiss_nn_dict[base_name] = dict()
            faiss_nn_dict[base_name][key] = nn_list

np.save('faiss/NN_dict_NA-Birds-small-zero-shot_{}.npy'.format(set), faiss_nn_dict)

