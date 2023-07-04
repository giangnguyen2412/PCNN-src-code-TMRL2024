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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
SAVE_FILE = "HAL9000.pt"

train_set = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)
validation_set = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="validation", transform=preprocess)

dataloaders = {
    "train": DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4),
    "validation": DataLoader(validation_set, batch_size=128, shuffle=True, num_workers=4),
}

print('train size: {}'.format(len(train_set)))
print('validation size: {}'.format(len(validation_set)))

dataset_sizes = {
    "train": len(train_set),
    "validation": len(validation_set),
}


model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

from collections import OrderedDict
new_ckpt = OrderedDict()
ckpt = torch.load('/home/giang/Downloads/advising_network/stanford-dogs/HAL9000.pt')
for k, v in ckpt.items():
    new_k = k.replace('module.', '')
    new_ckpt[new_k] = v

model.load_state_dict(new_ckpt)
print(model)

################################################################
