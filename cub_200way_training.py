import os
from params import RunningParams
RunningParams = RunningParams()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
TRAINING_FROM_SCRATCH = True
USE_INPUT_AS_EXPLANATION = False

RN18 = False
RN50 = True

PRETRAINED = True

if RN18 is True:
    if PRETRAINED is True:
        network = 'ImageNet RN18 pretrained'
    else:
        network = 'RN18 untrained'
elif RN50 is True:
    INAT = True
    IMAGENET = False
    if PRETRAINED is True:
        if INAT is True:
            network = 'INAT RN50 pretrained'
        else:
            network = 'ImageNet RN50 pretrained'
    else:
        network = 'RN50 untrained'

BLACK_EXPLANATION = False
USING_CLASS_EMBEDDING = RunningParams.USING_CLASS_EMBEDDING
if USING_CLASS_EMBEDDING is True:
    action = 'Using'
    FIXED = RunningParams.FIXED
    LEARNABLE = RunningParams.LEARNABLE

    if FIXED is True:
        optim = 'fixed'
    elif LEARNABLE is True:
        optim = 'learnable'
else:
    action = 'NOT Using'


if USING_CLASS_EMBEDDING is True:
    commit = "Run k={} for {}. {} {} class embedding and point-wise adding. I used OneLR scheduler".format(
        RunningParams.k_value, network, action, optim)
else:
    commit = "Run k={} for {}. {} class embedding and point-wise adding".format(
        RunningParams.k_value, network, action)

print(commit)
if len(commit) == 0:
    print("Please commit before running!")
    exit(-1)

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
import random
import tqdm
import wandb


assert RunningParams.CUB_TRAINING

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to("cpu").numpy()

train_dataset_transform = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_dataset_transform = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

training_folder = ImageFolderForNNs(
    root="/home/giang/Downloads/datasets/CUB/train1/",
    transform=train_dataset_transform,
)
train_loader = DataLoader(
    training_folder, batch_size=64, shuffle=True, num_workers=8, pin_memory=False
)

validation_folder = ImageFolderForNNs(
    root="/home/giang/Downloads/datasets/CUB/test0/", transform=train_dataset_transform
)
val_loader = DataLoader(
    validation_folder, batch_size=64, shuffle=False, num_workers=8, pin_memory=False
)

if TRAINING_FROM_SCRATCH is True:
    if RN18 is True:
        model = torchvision.models.resnet18(pretrained=PRETRAINED)
    elif RN50:
        model = torchvision.models.resnet50(pretrained=PRETRAINED)
        if INAT is True:
            model_dict = torch.load("/home/giang/Downloads/Cub-ResNet-iNat/resnet50_inat_pretrained_0.841.pth")
            model_dict = OrderedDict(
                {name.replace("layers.", ""): value for name, value in model_dict.items()}
            )
            model.load_state_dict(model_dict, strict=False)
    ################################################################
    if USING_CLASS_EMBEDDING is True:
        from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

        resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load(
            'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        resnet.load_state_dict(my_model_state_dict, strict=True)
        class_emd_matrix = resnet.classifier.weight

        if LEARNABLE is True:
            if RN18 is True:
                class_emd_matrix = class_emd_matrix.unsqueeze(dim=1)
                # Define the 1D average pooling layer
                avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
                learnable_emd_matrix = avg_pool(class_emd_matrix).squeeze().detach().clone()
                learnable_emd_matrix.requires_grad = True
            elif RN50 is True:
                learnable_emd_matrix = class_emd_matrix
        elif FIXED is True:
            if RN18 is True:
                class_emd_matrix = class_emd_matrix.unsqueeze(dim=1)
                # Define the 1D average pooling layer
                avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)
                class_emd_matrix = avg_pool(class_emd_matrix).squeeze()
            else:
                class_emd_matrix = class_emd_matrix
    #################################################################


# TODO: a) Dont need to use nn.Sequential and b) construct again the model because we may have two
#  FC layers in the model.
# Fortunately, we are doing feature_extractor then fc so we did not have error
if RN18 is True:
    fc = nn.Sequential(nn.Linear(512*(RunningParams.k_value + 1) + RunningParams.k_value, 512),
                   nn.ReLU(),
                   nn.Dropout(0.0),
                   nn.Linear(512, 200),
                   nn.Dropout(0.0)
                   ).cuda()
elif RN50 is True:
    fc = nn.Sequential(nn.Linear(2048 * (RunningParams.k_value + 1) + RunningParams.k_value, 512),
                       nn.ReLU(),
                       nn.Dropout(0.0),
                       nn.Linear(512, 200),
                       nn.Dropout(0.0)
                       ).cuda()


fc = nn.DataParallel(fc)


if TRAINING_FROM_SCRATCH is True:
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).cuda()
else:
    if PRETRAINED_CUB is True:
        feature_extractor = nn.Sequential(*list(model.children())[:-2]).cuda()
    else:
        feature_extractor = nn.Sequential(*list(model.children())[:-1]).cuda()

    model.fc = fc

feature_extractor = nn.DataParallel(feature_extractor)

criterion = nn.CrossEntropyLoss()

if TRAINING_FROM_SCRATCH is True:
    params = list(fc.parameters()) + list(feature_extractor.parameters())

    if USING_CLASS_EMBEDDING is True and LEARNABLE is True:
        params.append(learnable_emd_matrix)

    optimizer = optim.SGD([
        {'params': params, 'lr': RunningParams.learning_rate},
    ],
        momentum=0.9, weight_decay=5e-4)

    from torch.optim import lr_scheduler
    # Observe all parameters that are being optimized
    oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01,  # TODO: ask on pytorch forum how onelr_scheduler handles the lr of optimizer?
        steps_per_epoch=5994 // RunningParams.batch_size,
        epochs=RunningParams.epochs)

else:
    optimizer = optim.Adam(model.fc.parameters())


def test_model(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for _, (data, target, pths) in enumerate(val_loader):
            target = target.cuda()
            input_feat = feature_extractor(data[0].cuda()).squeeze()

            if RunningParams.k_value > 0:
                if USE_INPUT_AS_EXPLANATION is True and TRAINING_FROM_SCRATCH is True:
                    data[1] = torch.stack([data[0]] * 6, dim=1)
                explanations = data[1][:, 0:RunningParams.k_value, :, :, :]  # ignore 1st NN = query
                # Change the explanations to completely black images
                if BLACK_EXPLANATION is True:
                    explanations = torch.zeros_like(explanations)

                explanations_l = []
                sep_token = torch.zeros([RunningParams.k_value, 1, 1, 1], requires_grad=False).cuda()
                for sample_idx in range(explanations.shape[0]):
                    if USING_CLASS_EMBEDDING is True:
                        # Extract class embedding for prototypes
                        query_base_name = os.path.basename(pths[sample_idx])
                        # TODO: val_loader here
                        prototype_list = val_loader.dataset.faiss_nn_dict[query_base_name]
                        # TODO: Dont skip the first prototype in test
                        prototype_classes = [prototype_list[i].split('/')[-2] for i in range(0, RunningParams.k_value)]
                        prototype_class_ids = [val_loader.dataset.class_to_idx[c] for c in prototype_classes]
                        if FIXED is True:
                            class_embeddings = [class_emd_matrix[i] for i in prototype_class_ids]
                            class_embeddings = torch.stack(class_embeddings)
                            # Add two dims to fit with the conv features
                            class_embeddings = class_embeddings[:, :, None, None].cuda()
                            class_embeddings = class_embeddings.detach()
                        elif LEARNABLE is True:
                            one_hot_enc = F.one_hot(torch.tensor(prototype_class_ids), num_classes=200)
                            one_hot_enc = one_hot_enc.type(torch.FloatTensor).detach()
                            class_embeddings = torch.matmul(one_hot_enc, learnable_emd_matrix)
                            class_embeddings = class_embeddings[:, :, None, None].cuda()

                    explanation = explanations[sample_idx]
                    explanation_feat = feature_extractor(explanation.cuda())

                    # Point-wise multiplication prototypes and their class embeddings
                    if USING_CLASS_EMBEDDING is True:
                        explanation_feat = explanation_feat + class_embeddings

                    explanation_feat = torch.cat([sep_token, explanation_feat], dim=1)
                    squeezed_explanation_feat = explanation_feat.squeeze()
                    squeezed_explanation_feat = squeezed_explanation_feat.flatten()
                    explanations_l.append(squeezed_explanation_feat)

                explanation_feat = torch.stack(explanations_l)
                if BLACK_EXPLANATION is True:
                    explanation_feat = torch.zeros_like(explanation_feat)
                features = torch.cat((input_feat, explanation_feat), dim=1)
            else:
                features = input_feat

            outputs = fc(features)

            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * target.size(0)
            running_corrects += torch.sum(preds == target.data)

        epoch_loss = running_loss / len(validation_folder)
        epoch_acc = running_corrects.double()*100 / len(validation_folder)

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

        if RunningParams.k_value > 0:
            if USE_INPUT_AS_EXPLANATION is True and TRAINING_FROM_SCRATCH is True:
                data[1] = torch.stack([data[0]] * 6, dim=1)
            explanations = data[1][:, 1:RunningParams.k_value + 1, :, :, :]  # ignore 1st NN = query
            # Change the explanations to completely black images
            if BLACK_EXPLANATION is True:
                explanations = torch.zeros_like(explanations)

            explanations_l = []
            sep_token = torch.zeros([RunningParams.k_value, 1, 1, 1], requires_grad=False).cuda()
            for sample_idx in range(explanations.shape[0]):
                if USING_CLASS_EMBEDDING is True:
                    # Extract class embedding for prototypes
                    query_base_name = os.path.basename(pths[sample_idx])
                    # TODO: val_loader if in test
                    prototype_list = train_loader.dataset.faiss_nn_dict[query_base_name]
                    # TODO: Dont skip the first prototype in test
                    prototype_classes = [prototype_list[i].split('/')[-2] for i in range(1, RunningParams.k_value+1)]
                    prototype_class_ids = [train_loader.dataset.class_to_idx[c] for c in prototype_classes]
                    if FIXED is True:
                        class_embeddings = [class_emd_matrix[i] for i in prototype_class_ids]
                        class_embeddings = torch.stack(class_embeddings)
                        # Add two dims to fit with the conv features
                        class_embeddings = class_embeddings[:, :, None, None].cuda()
                        class_embeddings = class_embeddings.detach()
                    elif LEARNABLE is True:
                        one_hot_enc = F.one_hot(torch.tensor(prototype_class_ids), num_classes=200)
                        one_hot_enc = one_hot_enc.type(torch.FloatTensor).detach()
                        class_embeddings = torch.matmul(one_hot_enc, learnable_emd_matrix)
                        class_embeddings = class_embeddings[:, :, None, None].cuda()

                explanation = explanations[sample_idx]
                explanation_feat = feature_extractor(explanation.cuda())

                # Point-wise multiplication prototypes and their class embeddings
                if USING_CLASS_EMBEDDING is True:
                    explanation_feat = explanation_feat + class_embeddings

                explanation_feat = torch.cat([sep_token, explanation_feat], dim=1)
                squeezed_explanation_feat = explanation_feat.squeeze()
                squeezed_explanation_feat = squeezed_explanation_feat.flatten()
                explanations_l.append(squeezed_explanation_feat)

            explanation_feat = torch.stack(explanations_l)
            if BLACK_EXPLANATION is True:
                explanation_feat = torch.zeros_like(explanation_feat)
            features = torch.cat((input_feat, explanation_feat), dim=1)
        else:
            features = input_feat
        outputs = fc(features)

        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        oneLR_scheduler.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * target.size(0)
        running_corrects += torch.sum(preds == target.data)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double()*100 / len(training_folder)
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
    config=None,
    notes=commit

)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('cub_200way_training.py', policy='now')
wandb.save('datasets.py', policy='now')

best_acc = 0.0
num_epochs = RunningParams.epochs
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    model_trained = train_model(model, criterion, optimizer, num_epochs=10)
    val_acc = test_model(model_trained)

    # Clear GPU memory
    torch.cuda.empty_cache()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_trained.state_dict(), "./CUB_200way_best_model.pth")
    print("{} - Best accuracy: {:.4f}".format(wandb.run.name, best_acc))

wandb.finish()