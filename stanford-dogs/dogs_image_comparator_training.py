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
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from torchvision import datasets, models, transforms
from transformer import *
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

RunningParams = RunningParams()
Dataset = Dataset()

if [RunningParams.DOGS_TRAINING, RunningParams.CUB_TRAINING, RunningParams.CARS_TRAINING].count(True) > 1:
    print("There are more than one training datasets chosen, skipping training!!!")
    exit(-1)

train_dataset = RunningParams.aug_data_dir

image_datasets = dict()
image_datasets['train'] = ImageFolderForNNs(train_dataset, Dataset.data_transforms['train'])

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

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

                if RunningParams.VisionTransformer is True:
                    for param in model.module.feature_extractor.parameters():
                        param.requires_grad_(trainable) # frozen#TODO
                else:
                    if RunningParams.TRANSFORMER_ARCH:
                        for param in model.module.transformer_feat_embedder.parameters():
                            param.requires_grad_(trainable)

                        for param in model.module.transformer.parameters():
                            param.requires_grad_(trainable)

                        for param in model.module.cross_transformer.parameters():
                            param.requires_grad_(trainable)

                    for param in model.module.conv_layers.parameters():
                        param.requires_grad_(trainable)

                for param in model.module.branch3.parameters():
                    param.requires_grad_(trainable)

                for param in model.module.agg_branch.parameters():
                    param.requires_grad_(trainable)

            else:
                shuffle = False
                model.eval()  # Evaluation mode
                drop_last = False

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
                if RunningParams.XAI_method == RunningParams.NNs:
                    x = data[0].cuda()
                else:
                    x = data.cuda()

                labels = data[2].cuda()

                if (sum(labels)/RunningParams.batch_size) > 0.7 or (sum(labels)/RunningParams.batch_size) < 0.3:
                    print("Warning: The distribution of positive and negative in this batch is highly skewed."
                          "Beware of the loss function!")

                #####################################################

                explanation = data[1]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # data[-1] is the trivial augmented data
                    # breakpoint()
                    output, query, nns, emb_cos_sim = model(images=data[-1], explanations=explanation, scores=None)

                    p = torch.sigmoid(output)

                    # classify inputs as 0 or 1 based on the threshold of 0.5
                    preds = (p >= 0.5).long().squeeze()

                    loss = loss_func(output.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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
                {'{}_precision'.format(phase): precision, '{}_recall'.format(phase): recall, '{}_f1'.format(phase): f1})

            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, precision, recall, f1))

            ################################################################
            if scheduler is not None:
                scheduler.step()

            wandb.log({'{}_accuracy'.format(phase): epoch_acc * 100, '{}_loss'.format(phase): epoch_loss})

            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, yes_ratio.item() * 100, true_ratio.item() * 100))

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
                    'best_f1':best_f1,
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

if RunningParams.VisionTransformer is True:
    MODEL2 = ViT_AdvisingNetwork()
else:
    if RunningParams.TRANSFORMER_ARCH == True:
        MODEL2 = Transformer_AdvisingNetwork()
    else:
        MODEL2 = CNN_AdvisingNetwork()

CONTINUE_TRAINING = False
if CONTINUE_TRAINING is True:
    if RunningParams.resnet == 18:
        model_name = 'best_model_copper-moon-3322.pt'
    elif RunningParams.resnet == 34:
        model_name = 'best_model_woven-deluge-3324.pt'
    elif RunningParams.resnet == 50:
        model_name = 'best_model_dainty-blaze-3325.pt'

    model_path = os.path.join(RunningParams.prj_dir, 'best_models', model_name)

    MODEL2 = Transformer_AdvisingNetwork()
    MODEL2 = nn.DataParallel(MODEL2).cuda()

    checkpoint = torch.load(model_path)
    MODEL2.load_state_dict(checkpoint['model_state_dict'])
    MODEL2 = MODEL2.module

criterion = nn.BCEWithLogitsLoss().cuda()

# Observe all parameters that are being optimized
if RunningParams.VisionTransformer is True:
    optimizer_ft = optim.SGD([
        {'params': MODEL2.feature_extractor.parameters(), 'lr': 3e-4},
        {'params': MODEL2.branch3.parameters(), 'lr': 1e-2},
        {'params': MODEL2.agg_branch.parameters(), 'lr': 1e-2}
    ], momentum=0.9)
    oneLR_scheduler = None
else:
    optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

    max_lr = RunningParams.learning_rate * 10
    oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_ft, max_lr=max_lr,
        steps_per_epoch=dataset_sizes['train'] // RunningParams.batch_size,
        epochs=RunningParams.epochs)

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

config = {"train": train_dataset,
          "train_size": dataset_sizes['train'],
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'explanation': RunningParams.XAI_method,
          'k_value': RunningParams.k_value,
          'conv_layer': RunningParams.conv_layer,
          'HIGHPERFORMANCE_FEATURE_EXTRACTOR': RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR,
          'BOTTLENECK': RunningParams.BOTTLENECK,
          'trivial_augment_p': RunningParams.trivial_aument_p,
          }

print(config)

# if RunningParams.wandb_sess_name is not None:
#     wandb.init(
#         project="advising-network",
#         entity="luulinh90s",
#         config=config,
#         name=RunningParams.wandb_sess_name,
#     )
# else:
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config,
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save(f'{RunningParams.prj_dir}/params.py', policy='now')
wandb.save(f'{RunningParams.prj_dir}/datasets.py', policy='now')
wandb.save('dogs_image_comparator_training.py', policy='now')
wandb.save(f'{RunningParams.prj_dir}/transformer.py', policy='now')

if RunningParams.VisionTransformer is True:
    _, best_acc = train_model(
        MODEL2,
        criterion,
        optimizer_ft,
        None,
        config["num_epochs"])
else:
    _, best_acc = train_model(
        MODEL2,
        criterion,
        optimizer_ft,
        oneLR_scheduler,
        config["num_epochs"])

wandb.finish()