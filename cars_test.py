import math
import time
import os
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from datasets import StanfordDogsDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision import datasets, models, transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                         ])

DEVICE = torch.device("cuda")

#
train_dataset = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/BACKUP/train'
val_dataset = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/test'

from datasets import Dataset, StanfordDogsDataset, ImageFolderForNNs

train_set = ImageFolderForNNs(train_dataset, preprocess)
validation_set = ImageFolderForNNs(val_dataset, preprocess)


dataloaders = {
    "train": DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4),
    "validation": DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=4),
}

torch.manual_seed(42)


print('train size: {}'.format(len(train_set)))
print('validation size: {}'.format(len(validation_set)))

dataset_sizes = {
    "train": len(train_set),
    "validation": len(validation_set),
}


import torchvision

model = torchvision.models.resnet34(pretrained=True).cuda()
model.fc = nn.Linear(model.fc.in_features, 196)

my_model_state_dict = torch.load(
    '/home/giang/Downloads/advising_network/PyTorch-Stanford-Cars-Baselines/model_best_rn34.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(my_model_state_dict['state_dict'], strict=True)
model.cuda()
model.eval()


running_corrects = 0
data_loader = dataloaders['validation']

preds_dict = {}
# Iterate over data.
for inputs, labels, pths in data_loader:
    inputs = inputs[0].to(DEVICE)
    labels = labels.to(DEVICE)

    # track history if only in train
    outputs = model(inputs)
    model1_p = torch.nn.functional.softmax(outputs, dim=1)
    confs, preds = torch.max(model1_p, 1)

    for sample_idx in range(inputs.shape[0]):

        ############################################################################################
        img_name = os.path.basename(pths[sample_idx])
        correctness = (preds[sample_idx] == labels.data[sample_idx])
        preds_dict[img_name] = {}
        preds_dict[img_name]['correctness'] = correctness.item()
        preds_dict[img_name]['prediction'] = preds[sample_idx].item()
        preds_dict[img_name]['groundtruth'] = labels.data[sample_idx].item()
        preds_dict[img_name]['confidence'] = confs[sample_idx].item()
        ############################################################################################

    # statistics
    running_corrects += torch.sum(preds == labels.data)

epoch_acc = running_corrects.double() / dataset_sizes['validation'] * 100
print(epoch_acc)
################################################################
import numpy as np
np.save('SCars_preds_dict.npy', preds_dict)