# Training CUB-200 way classifiers using standard training
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import random

random.seed(42)

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to("cpu").numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

train_dataset_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_dataset_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

training_folder = ImageFolder(
    root="/home/anonymous/Downloads/datasets/CUB/train1/",
    transform=train_dataset_transform,
)
train_loader = DataLoader(
    training_folder, batch_size=768, shuffle=True, num_workers=32, pin_memory=True
)

validation_folder = ImageFolder(
    root="/home/anonymous/Downloads/datasets/CUB/test0/", transform=train_dataset_transform
)
val_loader = DataLoader(
    validation_folder, batch_size=768, shuffle=False, num_workers=8, pin_memory=False
)

model = torchvision.models.resnet34(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(512, 200)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

def test_model(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for _, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * target.size(0)
        running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(validation_folder)
    epoch_acc = running_corrects.double() / len(validation_folder)

    print("-" * 10)
    print("loss: {:.4f}, acc: {:.4f}".format(epoch_loss, epoch_acc))

def train_model(model, criterion, optimizer, num_epochs=3):

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_corrects = 0

        for _, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)

            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * target.size(0)
            running_corrects += torch.sum(preds == target.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(training_folder)

        if epoch % 10 == 0:
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 10)
            print("loss: {:.4f}, acc: {:.4f}".format(epoch_loss, epoch_acc))

    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=50)

test_model(model_trained)

torch.save(model_trained.state_dict(), "./imagenet_pretrained_resnet34_cub_200way.pth")