# Training script for CUB AdvNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy
import wandb
import statistics
import pdb
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from torchvision import datasets, models, transforms
from transformer import *
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions

# torch.backends.cudnn.benchmark = True
# plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

RunningParams = RunningParams('CARS')

Dataset = Dataset()

import torchvision

if RunningParams.resnet == 50:
    model = torchvision.models.resnet50(pretrained=True).cuda()
elif RunningParams.resnet == 34:
    model = torchvision.models.resnet34(pretrained=True).cuda()
elif RunningParams.resnet == 18:
    model = torchvision.models.resnet18(pretrained=True).cuda()
model.fc = nn.Linear(model.fc.in_features, 196)

my_model_state_dict = torch.load(
        f'{RunningParams.prj_dir}/pretrained_models/cars-196/model_best_rn{RunningParams.resnet}.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(my_model_state_dict['state_dict'], strict=True)
model.eval()

MODEL1 = model.cuda()

fc = MODEL1.fc
fc = fc.cuda()

train_dataset = RunningParams.aug_data_dir

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


image_datasets = dict()
image_datasets['train'] = ImageFolderForNNs(train_dataset, data_transform)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)


def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                shuffle = True
                model.train()  # Training mode
                drop_last = True
                frozen = False
                trainable = True

                for param in model.module.conv_layers.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.transformer_feat_embedder.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.transformer.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.cross_transformer.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.branch3.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.agg_branch.parameters():
                    param.requires_grad_(trainable)

            else:
                shuffle = False
                drop_last = False
                model.eval()  # Evaluation mode

            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=RunningParams.batch_size,
                shuffle=shuffle,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
                drop_last=drop_last
            )

            running_loss = 0.0
            running_corrects = 0

            yes_cnt = 0
            true_cnt = 0

            labels_val = []
            preds_val = []

            for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
                x = data[0].cuda()

                gts = gt.cuda()

                embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18

                out = fc(embeddings)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                model2_gt = (predicted_ids == gts) * 1  # 0 and 1

                # Get the label (0/1) from the faiss npy file
                labels = data[2].cuda()

                if (sum(labels) / RunningParams.batch_size) > 0.7 or (sum(labels) / RunningParams.batch_size) < 0.3:
                    print("Warning: The distribution of positive and negative in this batch is high skewed."
                          "Beware of the loss function!")

                #####################################################
                explanation = data[1]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # data[-1] is the trivial augmented data
                    output, query, nns, emb_cos_sim = model(images=data[-1], explanations=explanation, scores=score)

                    p = torch.sigmoid(output)

                    # classify inputs as 0 or 1 based on the threshold of 0.5
                    preds = (p >= 0.5).long().squeeze()
                    loss = loss_func(output.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # if phase == 'val':
                if True:
                    preds_val.append(preds)
                    labels_val.append(labels)

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == labels.data)

                yes_cnt += sum(preds)
                true_cnt += sum(labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            yes_ratio = yes_cnt.double() / len(image_datasets[phase])
            true_ratio = true_cnt.double() / len(image_datasets[phase])

            ################################################################

            # Calculate precision, recall, and F1 score
            preds_val = torch.cat(preds_val, dim=0)
            labels_val = torch.cat(labels_val, dim=0)

            precision = precision_score(labels_val.cpu(), preds_val.cpu())
            recall = recall_score(labels_val.cpu(), preds_val.cpu())
            f1 = f1_score(labels_val.cpu(), preds_val.cpu())
            confusion_matrix_ = confusion_matrix(labels_val.cpu(), preds_val.cpu())
            print(confusion_matrix_)

            wandb.log(
                {'{}_precision'.format(phase): precision, '{}_recall'.format(phase): recall,
                 '{}_f1'.format(phase): f1})

            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, precision, recall, f1))

            ################################################################

            scheduler.step()

            wandb.log({'{}_accuracy'.format(phase): epoch_acc * 100, '{}_loss'.format(phase): epoch_loss})
            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, yes_ratio.item() * 100,
                                                   true_ratio.item() * 100))
            # deep copy the model
            if f1 >= best_f1:
                best_f1 = f1
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                ckpt_path = '{}/best_models/best_model_{}.pt' \
                    .format(RunningParams.prj_dir, wandb.run.name)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_loss,
                    'best_loss': best_loss,
                    'best_f1': best_f1,
                    'val_acc': epoch_acc * 100,
                    'val_yes_ratio': yes_ratio * 100,
                    'running_params': RunningParams,
                }, ckpt_path)
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


MODEL2 = Transformer_AdvisingNetwork()

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

criterion = nn.BCEWithLogitsLoss().cuda()

# Observe all parameters that are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

max_lr = RunningParams.learning_rate * 10
oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_ft, max_lr=max_lr,
    steps_per_epoch=dataset_sizes['train'] // RunningParams.batch_size,
    epochs=RunningParams.epochs)

config = {"train": train_dataset,
          "train_size": dataset_sizes['train'],
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'k_value': RunningParams.k_value,
          'conv_layer': RunningParams.conv_layer,
          }

print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config,
)

wandb.save(f'{RunningParams.prj_dir}/params.py')
wandb.save(f'{RunningParams.prj_dir}/datasets.py')
wandb.save('car_image_comparator_training.py')
wandb.save(f'{RunningParams.prj_dir}/transformer.py')

_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])


wandb.finish()