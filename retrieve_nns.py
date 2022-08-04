import os
import numpy as np
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import torchvision.transforms as transforms
import wandb
import random
from PIL import Image
from operator import itemgetter
import faiss
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer

from datasets import Dataset
from params import RunningParams
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

Dataset = Dataset()
RunningParams = RunningParams()


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        data = original_tuple[0]
        label = original_tuple[1]
        if data.shape[0] == 1:
            print('gray images')
            data = torch.cat([data, data, data], dim=0)

        # make a new tuple that includes original and the path
        tuple_with_path = (data, label, path)
        return tuple_with_path


imagenet_train_path = '/home/giang/Downloads/datasets/imagenet1k-val'
model = torchvision.models.resnet18(pretrained=True).eval()
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

in_features = model.fc.in_features
k_value = 5

if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

    dataset = ImageFolderWithPaths(imagenet_train_path, Dataset.data_transforms['train'])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=RunningParams.batch_size,
        shuffle=False,  # no need shuffle
        num_workers=8,
        pin_memory=True,
    )

    stack_embeddings = []
    stack_labels = []
    gallery_paths = []

    for batch_idx, (data, label, path) in enumerate(tqdm(data_loader)):
        embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
        embeddings = torch.flatten(embeddings, start_dim=1)
        # print(embeddings.shape)
        # TODO: add data tensors here to return data using index
        # TODO: search in range only
        # TODO: Move everything to GPU
        stack_embeddings.append(embeddings.cpu().detach().numpy())
        stack_labels.append(label.cpu().detach().numpy())
        gallery_paths.extend(path)

    stack_embeddings = np.concatenate(stack_embeddings, axis=0)
    stack_labels = np.concatenate(stack_labels, axis=0)
    # stack_embeddings = stack_embeddings.cpu().detach().numpy()

    descriptors = np.vstack(stack_embeddings)
    cpu_index = faiss.IndexFlatL2(in_features)
    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )
    print(gpu_index.ntotal)

    gpu_index.add(descriptors)

    for batch_idx, (data, label, path) in enumerate(tqdm(data_loader)):
        embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
        embeddings = torch.flatten(embeddings, start_dim=1)
        embeddings = embeddings.cpu().detach().numpy()
        distance, indices = gpu_index.search(embeddings, k_value)
        # Tracer()()
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))



        for file_index, ax_i in zip(indices[0], np.array(ax).flatten()):
            ax_i.imshow(plt.imread(gallery_paths[file_index]))

        plt.show()
        break

    # query_image = 'Jewellery/bracelet/bracelet_048.jpg'
    # img = Image.open(query_image)
    # input_tensor = transforms_(img)
    # input_tensor = input_tensor.view(1, *input_tensor.shape)
    # with torch.no_grad():
    #     query_descriptors = pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
    #     distance, indices = index.search(query_descriptors.reshape(1, 2048), 9)