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
from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from models import MyCustomResnet18, AdvisingNetwork
from params import RunningParams
from helpers import HelperFunctions

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

data_dir = '/home/giang/Downloads/advising_net_training/'
virtual_train_dataset = '{}/train'.format(data_dir)

train_dataset = '/home/giang/Downloads/datasets/random_train_dataset'
val_dataset = '/home/giang/Downloads/datasets/imagenet1k-val'

if not HelperFunctions.is_running(os.path.basename(__file__)):
    print('Creating symlink datasets ...')
    if os.path.islink(virtual_train_dataset) is True:
        os.unlink(virtual_train_dataset)
    os.symlink(train_dataset, virtual_train_dataset)

    virtual_val_dataset = '{}/val'.format(data_dir)
    if os.path.islink(virtual_val_dataset) is True:
        os.unlink(virtual_val_dataset)
    os.symlink(val_dataset, virtual_val_dataset)
else:
    print('Script is running! No creating symlink datasets!')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

RunningParams = RunningParams()
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'val']}


for (name, ds) in image_datasets.items():
    writer = DatasetWriter(f'ffcv_output/imagenet_{name}.beton', {
        'image': RGBImageField(write_mode='proportion',
                               max_resolution=500,
                               compress_probability=0.5,
                               jpeg_quality=90),
        'label': IntField(),
    },
        num_workers=16)
    writer.from_indexed_dataset(ds)


