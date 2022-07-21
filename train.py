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

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, Convert, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.transforms import *
import torchvision as tv

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

RunningParams = RunningParams()

model1_name = 'resnet18'
MODEL1 = models.resnet18(pretrained=True).eval().cuda()

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

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          RunningParams.data_transforms[x])
                  for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
pdist = nn.PairwiseDistance(p=2)



IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if RunningParams.FFCV_loader is True:
                data_loader = Loader('ffcv_output/imagenet_{}.beton'.format(phase),
                                     batch_size=RunningParams.batch_size,
                                     num_workers=8, order=OrderOption.RANDOM,
                                     drop_last=True,
                                     pipelines={'image': [
                                      RandomResizedCropRGBImageDecoder((224, 224)),
                                      RandomHorizontalFlip(),
                                      ToTensor(),
                                      ToDevice(torch.device('cuda:0'), non_blocking=True),
                                      ToTorchImage(),
                                      # Standard torchvision transforms still work!
                                      NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
                                     ], 'label':
                                     [
                                        IntDecoder(),
                                        ToTensor(),
                                        Squeeze(),
                                        ToDevice(torch.device('cuda:0'), non_blocking=True),
                                            ]}
                                     )
            else:
                data_loader = torch.utils.data.DataLoader(
                    image_datasets[phase],
                    batch_size=RunningParams.batch_size,
                    shuffle=True,  # turn shuffle to True
                    num_workers=8,
                    pin_memory=True,
                )

            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()  # Evaluation mode

            running_loss = 0.0
            running_corrects = 0

            yes_cnt = 0
            true_cnt = 0

            for batch_idx, (data, gt) in enumerate(tqdm(data_loader)):
                x = data.cuda()
                gts = gt.cuda()

                out = MODEL1(x)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                model2_gt = (predicted_ids == gts) * 1  # 0 and 1

                # TODO: Incorporate confidence score: Correct = confidence score, Wrong = 1 confidence score
                # for gt_idx in range(len(model2_gt)):
                #     if model2_gt[gt_idx].item() == 1:
                #         model2_gt[gt_idx] = [1 - score[gt_idx].detach().item(), score[gt_idx].detach().item()]
                #     else:
                #         model2_gt[gt_idx] = [score[gt_idx].detach().item(), 1 - score[gt_idx].detach().item()]

                saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)
                explanation = torch.amax(saliency, dim=(1, 2, 3,))  # normalize the heatmaps
                explanation = torch.div(saliency, explanation.reshape(data.shape[0], 1, 1, 1))  # scale to 0->1

                # Incorporate explanations
                if RunningParams.XAI_method == 'No-XAI':
                    inputs = x
                elif RunningParams.XAI_method == 'GradCAM':
                    inputs = explanation * x  # multiply query & heatmap

                labels = model2_gt

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if RunningParams.advising_network is True:
                        output = model(x, saliency, model1_p)
                        p = torch.nn.functional.softmax(output, dim=1)
                        _, preds = torch.max(p, 1)
                        loss = criterion(p, labels)
                        # why val loss is so slow? debug the p and preds here

                    else:
                        ds_output, fc_output = model(inputs)
                        # print(fc_output[0][0])
                        model2_p = torch.nn.functional.softmax(fc_output, dim=1)
                        p = torch.nn.functional.softmax(ds_output, dim=1)
                        _, preds = torch.max(p, 1)

                        p2 = torch.nn.functional.softmax(fc_output, dim=1)
                        score2, _ = torch.topk(p2, 1, dim=1)

                        if RunningParams.top1 is True:
                            confidence_loss = pdist(score.squeeze(), score2.squeeze())
                        else:
                            confidence_loss = pdist(model2_p, model1_p).mean()

                        label_loss = criterion(p, labels)
                        # If true --> penalize the old softmax -> 1 at the place, all other 0
                        # If wrong --> penalize the top-1 -> 0 at the place, all other
                        # TODO: add criterion to penalize the softmax score
                        if RunningParams.confidence_loss is True:
                            loss = 0.5 * confidence_loss + 0.5 * label_loss
                        else:
                            loss = label_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                yes_cnt += sum(preds)
                true_cnt += sum(labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            yes_ratio = yes_cnt.double() / len(image_datasets[phase])
            true_ratio = true_cnt.double() / len(image_datasets[phase])

            wandb.log({'{}_accuracy'.format(phase): epoch_acc, '{}_loss'.format(phase): epoch_loss})

            print('{} - Loss: {:.4f} Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                phase, epoch_loss, epoch_acc*100, yes_ratio * 100, true_ratio*100))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                ckpt_path = '/home/giang/Downloads/advising_net_training/best_models/best_model_{}.pt'\
                    .format(wandb.run.name)

                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_loss,
                    'val_acc': epoch_acc*100,
                }, ckpt_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


if RunningParams.advising_network is True:
    model2_name = 'AdvisingNetwork'
    MODEL2 = AdvisingNetwork()
else:
    model2_name = 'MyCustomResnet18'
    MODEL2 = MyCustomResnet18(pretrained=True, fine_tune=RunningParams.fine_tune)

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_ft, max_lr=0.01,
    steps_per_epoch=dataset_sizes['train']//RunningParams.batch_size,
    epochs=RunningParams.epochs)

config = {"train": train_dataset,
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "model1": model1_name,
          "model2": model2_name,
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'explanation': RunningParams.XAI_method,
          'confidence_loss': RunningParams.confidence_loss,
          'top1': RunningParams.top1,
          'fine_tuning': RunningParams.fine_tune,
          'advising_net': RunningParams.advising_network,
          'query_frozen': RunningParams.query_frozen,
          'heatmap_frozen': RunningParams.heatmap_frozen,
          }


print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config
)

_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])
wandb.finish()
