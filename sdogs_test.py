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

def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return T.Compose([
        T.Resize((height, width)),
        T.Pad(pad_values),
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)


DEVICE = torch.device("cuda")

# train_set = StanfordDogsDataset(
#     root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)
# validation_set = StanfordDogsDataset(
#     root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="test", transform=preprocess)


#
train_dataset = '/home/giang/Downloads/advising_network/stanford-dogs/data/images/BACKUP/train'
val_dataset = '/home/giang/Downloads/advising_network/stanford-dogs/data/images/test'

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

model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

from collections import OrderedDict
new_ckpt = OrderedDict()
ckpt = torch.load('/home/giang/Downloads/advising_network/stanford-dogs/HAL9002_RN34.pt')
print(ckpt['val_acc']/100)

for k, v in ckpt['model_state_dict'].items():
    new_k = k.replace('module.', '')
    new_ckpt[new_k] = v

model.load_state_dict(new_ckpt)
model.cuda()
model.eval()

# Because it has a unique type of mapping the labels, we need reference_dataset to convert the predicted ids (StanfordDogsDataset) to ImageFolder ids
reference_dataset = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)


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

        ################################ CONVERT THE PREDICTED IDS #################################
        predicted_idx = preds[sample_idx].item()
        dog_name = reference_dataset.mapping[predicted_idx]
        preds[sample_idx] = data_loader.dataset.class_to_idx[dog_name]
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
np.save('SDogs_preds_dict.npy', preds_dict)