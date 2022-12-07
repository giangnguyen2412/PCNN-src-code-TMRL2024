import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import os
import pickle
import time
from collections import OrderedDict

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
from datasets import ImageFolderForNNs
from params import RunningParams
import random
import tqdm
import wandb


RunningParams = RunningParams()

random.seed(42)

for i in range(2000):
    random.randint(0, 999)

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to("cpu").numpy()


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

training_folder = ImageFolderForNNs(
    root="/home/giang/Downloads/datasets/CUB/train1/",
    transform=train_dataset_transform,
)
train_loader = DataLoader(
    training_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=True
)

validation_folder = ImageFolderForNNs(
    root="/home/giang/Downloads/datasets/CUB/test0/", transform=train_dataset_transform
)
val_loader = DataLoader(
    validation_folder, batch_size=512, shuffle=False, num_workers=8, pin_memory=False
)

PRETRAINED_CUB = True
if PRETRAINED_CUB is True:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    model_dict = torch.load(
        'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    model.load_state_dict(model_dict, strict=True)

else:

    model = torchvision.models.resnet50(pretrained=True).cuda()
    model_dict = torch.load("/home/giang/Downloads/Cub-ResNet-iNat/resnet50_inat_pretrained_0.841.pth")
    model_dict = OrderedDict(
        {name.replace("layers.", ""): value for name, value in model_dict.items()}
    )
    model.load_state_dict(model_dict, strict=False)

for param in model.parameters():
    param.requires_grad = False

# TODO: a) Dont need to use nn.Sequential and b) construct again the model because we may have two Linear layer in the model.
# Fortunately, we are doing feature_extractor then fc so we did not have error
model.fc = nn.Sequential(nn.Linear(2048*(RunningParams.k_value+1), 200)).cuda()

if PRETRAINED_CUB is True:
    feature_extractor = nn.Sequential(*list(model.children())[:-2]).cuda()
else:
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).cuda()
# feature_extractor = nn.DataParallel(feature_extractor)
fc = model.fc
# fc = nn.DataParallel(fc)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


def test_model(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for _, (data, target, pths) in enumerate(val_loader):
        target = target.cuda()

        input_feat = feature_extractor(data[0].cuda()).squeeze()

        explanations = data[1][:, 0:RunningParams.k_value, :, :, :].cuda()  # ignore 1st NN = query
        explanations_l = []
        for sample_idx in range(explanations.shape[0]):
            explanation = explanations[sample_idx]
            explanation_feat = feature_extractor(explanation)
            squeezed_explanation_feat = explanation_feat.squeeze()
            squeezed_explanation_feat = squeezed_explanation_feat.flatten()
            explanations_l.append(squeezed_explanation_feat)

        explanation_feat = torch.stack(explanations_l)

        features = torch.cat((input_feat, explanation_feat), dim=1)
        outputs = fc(features)

        loss = criterion(outputs, target)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * target.size(0)
        running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(validation_folder)
    epoch_acc = running_corrects.double() / len(validation_folder)

    phase = "val"
    wandb.log({'{}_accuracy'.format(phase): epoch_acc, '{}_loss'.format(phase): epoch_loss})

    print("-" * 10)
    print("Testing loss: {:.4f}, acc: {:.4f}".format(epoch_loss, epoch_acc))
    return epoch_acc


def train_model(model, criterion, optimizer, num_epochs=3):

    model.train()

    # for epoch in range(num_epochs):

    running_loss = 0.0
    running_corrects = 0

    for _, (data, target, pths) in enumerate(train_loader):
        target = target.cuda()
        input_feat = feature_extractor(data[0].cuda()).squeeze()

        explanations = data[1][:, 1:RunningParams.k_value + 1, :, :, :].cuda()  # ignore 1st NN = query
        explanations_l = []
        for sample_idx in range(explanations.shape[0]):
            explanation = explanations[sample_idx]
            explanation_feat = feature_extractor(explanation)
            squeezed_explanation_feat = explanation_feat.squeeze()
            squeezed_explanation_feat = squeezed_explanation_feat.flatten()
            explanations_l.append(squeezed_explanation_feat)

        explanation_feat = torch.stack(explanations_l)

        features = torch.cat((input_feat, explanation_feat), dim=1)
        outputs = fc(features)

        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * target.size(0)
        running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / len(training_folder)
    phase = "training"
    wandb.log({'{}_accuracy'.format(phase): epoch_acc, '{}_loss'.format(phase): epoch_loss})

    # if epoch % 10 == 0:
    if True:
        # print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        print("Training: loss: {:.4f}, acc: {:.4f}".format(epoch_loss, epoch_acc))

    return model

print(RunningParams.__dict__)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=None
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('cub_200way_training.py', policy='now')
wandb.save('datasets.py', policy='now')

best_acc = 0.0
num_epochs = 50
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    model_trained = train_model(model, criterion, optimizer, num_epochs=10)
    val_acc = test_model(model_trained)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_trained.state_dict(), "./CUB_200way_best_model.pth")
    print("Best accuracy: {:.4f}".format(best_acc))

wandb.finish()
