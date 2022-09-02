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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

MODEL1 = models.resnet18(pretrained=True).eval()
feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)
fc = MODEL1.fc
fc = fc.cuda()

Dataset = Dataset()
RunningParams = RunningParams()

RETRIEVE_TOP1_NEAREST = True


in_features = MODEL1.fc.in_features
print("Building FAISS index...")
faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/train', transform=Dataset.data_transforms['train'])
# faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/datasets/random_train_dataset_1k', transform=Dataset.data_transforms['train'])
faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if RETRIEVE_TOP1_NEAREST is True:
    INDEX_FILE = 'faiss/faiss_1M3_class_idx_dict.npy'
    if os.path.exists(INDEX_FILE):
        print("FAISS class index exists!")
        faiss_nns_class_dict = np.load('faiss/faiss_1M3_class_idx_dict.npy', allow_pickle="False", ).item()
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
        np.save('faiss/faiss_1M3_class_idx_dict.npy', faiss_nns_class_dict)
else:
    INDEX_FILE = 'faiss/faiss_1M3.index'
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

            # for sample_idx in range(input_data.shape[0]):
            #     base_name = os.path.basename(paths[sample_idx]) + '.pt'
            #     save_path = os.path.join('/home/giang/Downloads/datasets/processed_train', base_name)
            #     torch.save(input_data[sample_idx], save_path)

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
        faiss.write_index(faiss_cpu_index, 'faiss/faiss_1M3.index')


# data_dir = '/home/giang/Downloads/advising_net_training/'
data_dir = '/home/giang/Downloads/'

# print('Entries in faiss index: {}'.format(faiss_gpu_index.ntotal))
image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                      Dataset.data_transforms[x])
              for x in ['train']}

train_loader = torch.utils.data.DataLoader(
                image_datasets['train'],
                batch_size=RunningParams.batch_size,
                shuffle=False,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
                )

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

faiss_nn_dict = dict()
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    if RETRIEVE_TOP1_NEAREST is True:
        out = fc(embeddings)
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
    np.save('faiss/faiss_1M3_train_class_dict.npy', faiss_nn_dict)
else:
    np.save('faiss/faiss_1M3_dict.npy', faiss_nn_dict)
