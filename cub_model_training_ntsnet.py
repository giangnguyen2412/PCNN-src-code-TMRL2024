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

from tqdm import tqdm
from torchvision import datasets, models, transforms
from transformer import *
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

RunningParams = RunningParams()
Dataset = Dataset()

assert (RunningParams.DOGS_TRAINING is False and RunningParams.CARS_TRAINING is False)

if [RunningParams.DOGS_TRAINING, RunningParams.CUB_TRAINING, RunningParams.CARS_TRAINING].count(True) > 1:
    print("There are more than one training datasets chosen, skipping training!!!")
    exit(-1)

################################################################
import os
import torch.utils.data
from torch.nn import DataParallel
from core import model, dataset
from torch import nn
from tqdm import tqdm
net = model.attention_net(topN=6)
ckpt = torch.load(f'{RunningParams.parent_dir}/NTS-Net/model.ckpt')

net.load_state_dict(ckpt['net_state_dict'])

net.eval()
net = net.cuda()
net = DataParallel(net)
MODEL1 = net
MODEL1.eval()
################################################################

train_dataset = f'{RunningParams.parent_dir}/datasets/CUB/advnet/CUB_train_all_NTSNet'
val_dataset = f'{RunningParams.parent_dir}/datasets/CUB/advnet/val'
full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if RunningParams.XAI_method == RunningParams.NNs:
    image_datasets = dict()
    from PIL import Image
    data_transforms = transforms.Compose([
        transforms.Resize((600, 600), interpolation=Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_datasets['train'] = ImageFolderForNNs(train_dataset, data_transforms)
    image_datasets['val'] = ImageFolderForNNs(val_dataset, data_transforms)
else:
    pass
    # Not implemented

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                shuffle = True
                model.train()  # Training mode

                for param in model.module.conv_layers.parameters():
                    param.requires_grad_(True)

                for param in model.module.transformer_feat_embedder.parameters():
                    param.requires_grad_(True)

                for param in model.module.transformer.parameters():
                    param.requires_grad_(True)

                for param in model.module.cross_transformer.parameters():
                    param.requires_grad_(True)

                for param in model.module.branch3.parameters():
                    param.requires_grad_(True)

                for param in model.module.agg_branch.parameters():
                    param.requires_grad_(True)

            else:
                shuffle = False
                model.eval()  # Evaluation mode

            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=RunningParams.batch_size,
                shuffle=shuffle,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
            )

            running_loss = 0.0
            running_corrects = 0

            yes_cnt = 0
            true_cnt = 0

            for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
                if RunningParams.XAI_method == RunningParams.NNs:
                    x = data[0].cuda()
                else:
                    x = data.cuda()
                if len(data_loader.dataset.classes) < 200:
                    for sample_idx in range(x.shape[0]):
                        tgt = gt[sample_idx].item()
                        class_name = data_loader.dataset.classes[tgt]
                        id = full_cub_dataset.class_to_idx[class_name]
                        gt[sample_idx] = id

                gts = gt.cuda()

                _, out, _, _, _ = MODEL1(x)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                # MODEL1 Y/N label for input x
                model2_gt = (predicted_ids == gts) * 1  # 0 and 1
                labels = model2_gt

                if phase == 'train':
                    labels = data[2].cuda()

                #####################################################
                explanation = data[1]

                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        output, query, nns, emb_cos_sim = model(images=data[-1], explanations=explanation, scores=score)
                    else:
                        output, query, nns, emb_cos_sim = model(images=x, explanations=explanation, scores=score)

                    p = torch.sigmoid(output)

                    # classify inputs as 0 or 1 based on the threshold of 0.5
                    preds = (p >= 0.5).long().squeeze()

                    loss = loss_func(output.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == labels.data)

                yes_cnt += sum(preds)
                true_cnt += sum(labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            yes_ratio = yes_cnt.double() / len(image_datasets[phase])
            true_ratio = true_cnt.double() / len(image_datasets[phase])

            if phase == 'train':
                scheduler.step()

            wandb.log({'{}_accuracy'.format(phase): epoch_acc * 100, '{}_loss'.format(phase): epoch_loss})

            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, yes_ratio.item() * 100, true_ratio.item() * 100))

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
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
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'explanation': RunningParams.XAI_method,
          'k_value': RunningParams.k_value,
          'conv_layer': RunningParams.conv_layer,
          'HIGHPERFORMANCE_FEATURE_EXTRACTOR': RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR,
          'BOTTLENECK': RunningParams.BOTTLENECK,
          }

print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('datasets.py', policy='now')
wandb.save('cub_model_training.py', policy='now')
wandb.save('transformer.py', policy='now')

_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])

wandb.finish()