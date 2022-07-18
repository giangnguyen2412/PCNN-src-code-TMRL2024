from __future__ import print_function, division

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
from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from IPython.core.debugger import Tracer

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    # 'train': transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def random_but(n, end, start=0):
    # r = list(range(start, end))
    # r.remove(n)
    # return random.choice(r)
    if n == end-1:
        n = start
    else:
        n += 1
    return n


model1_name = 'resnet18'
MODEL1 = models.resnet18(pretrained=True).eval().cuda()
bs = 256
cam_RNlayer = 'layer4'
BATCH_BALANCE = False  # this is buggy, making nan output
FROZEN_FEATURE = False
XAI_method = 'No-XAI'

data_dir = '/home/giang/Downloads/advising_net_training/'
virtual_train_dataset = '{}/train'.format(data_dir)

if os.path.islink(virtual_train_dataset) is True:
    os.unlink(virtual_train_dataset)
train_dataset = '/home/giang/Downloads/datasets/random_train_dataset'
# train_dataset = '/home/giang/Downloads/train'
os.symlink(train_dataset, virtual_train_dataset)

virtual_val_dataset = '{}/val'.format(data_dir)
if os.path.islink(virtual_val_dataset) is True:
    os.unlink(virtual_val_dataset)
val_dataset = '/home/giang/Downloads/datasets/imagenet1k-val'
os.symlink(val_dataset, virtual_val_dataset)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=bs,
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

                if BATCH_BALANCE is True and phase == 'train':
                    model2_gt = torch.randint(0, 2, (data.shape[0],)).cuda()  # generate 0 and 1
                    # Random initialized tensor
                    index = torch.zeros([data.shape[0], 1], dtype=torch.long)

                    for gt_idx in range(len(model2_gt)):
                        if model2_gt[gt_idx].item() == 1:
                            index[gt_idx] = gts[gt_idx].item()
                        else:
                            index[gt_idx] = random_but(gts[gt_idx].item(), 1000, start=0)

                    index = index.cuda()
                else:
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

                saliency = grad_cam(MODEL1, x, index, saliency_layer=cam_RNlayer, resize=True)
                explanation = torch.amax(saliency, dim=(1, 2, 3,))  # normalize the heatmaps
                explanation = torch.div(saliency, explanation.reshape(data.shape[0], 1, 1, 1))  # scale to 0->1

                # Incorporate explanations
                if XAI_method == 'No-XAI':
                    inputs = x
                elif XAI_method == 'GradCAM':
                    inputs = explanation * x  # encoding query and heatmap,
                # I did not use softmax score yet.
                # Append after the fc the des layer of 2 units to incoporate softmax vectors
                # If I randomize to get the negative samples, the softmax scores do not
                # reflect model response of top1 (i.e. predicted label does not have the highest prob.)
                labels = model2_gt

                # inputs = inputs.cuda()
                # labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    ds_output, fc_output = model(inputs)
                    # print(fc_output[0][0])
                    model2_p = torch.nn.functional.softmax(fc_output, dim=1)
                    p = torch.nn.functional.softmax(ds_output, dim=1)
                    _, preds = torch.max(p, 1)

                    pdist = nn.PairwiseDistance(p=2)
                    confidence_loss = pdist(model2_p, model1_p).mean()
                    label_loss = criterion(p, labels)
                    # If true --> penalize the old softmax -> 1 at the place, all other 0
                    # If wrong --> penalize the top-1 -> 0 at the place, all other
                    # TODO: add criterion to penalize the softmax score
                    loss = 0.0*confidence_loss + 1.0*label_loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print('{} - Batch Yes ratio: {:.2f}'.format(phase, (preds.sum()*100) / data.shape[0]))
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

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.f\%}')
            print('{} - Loss: {:.4f} Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                phase, epoch_loss, epoch_acc*100, yes_ratio * 100, true_ratio*100))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


class MyCustomResnet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet18 = models.resnet18(pretrained=pretrained)
        if FROZEN_FEATURE:
            for param in resnet18.parameters():
                param.requires_grad = False

        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = nn.Sequential(*self.features)
        in_features = resnet18.fc.in_features
        self.fc0 = nn.Linear(in_features, 1000)
        self.fc0_bn = nn.BatchNorm1d(1000, eps=1e-2)
        self.fc1 = nn.Linear(1000, 2)
        self.fc1_bn = nn.BatchNorm1d(2, eps=1e-2)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        fc_output = self.fc0_bn(torch.nn.functional.relu(self.fc0(output)))
        ds_output = self.fc1_bn(torch.nn.functional.relu(self.fc1(fc_output)))

        # fc_output = self.fc0(output)
        # ds_output = self.fc1(fc_output)
        return ds_output, fc_output


model2_name = 'MyCustomResnet18'
model_ft = MyCustomResnet18()


# model_ft = models.resnet18(pretrained=True)


# num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)

MODEL2 = model_ft.cuda()
MODEL2 = nn.DataParallel(MODEL2)

criterion = nn.CrossEntropyLoss()

lr = 0.001
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

epochs = 25

oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.01, steps_per_epoch=
                                                      dataset_sizes['train']//bs, epochs=epochs)

config = {"train": train_dataset,
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "model1": model1_name,
          "model2": model2_name,
          "num_epochs": epochs,
          "batch_size": bs,
          "batch_balance": BATCH_BALANCE,
          "frozen_feat": FROZEN_FEATURE,
          "learning_rate": lr,
          'explanation': XAI_method,
          }


print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config
)

model_ft, best_acc = train_model(MODEL2, criterion, optimizer_ft, oneLR_scheduler,
                       config["num_epochs"])

torch.save(model_ft, '/home/giang/Downloads/advising_net_training/models/best_model_{}_{}.pth'.format(wandb.run.name, best_acc))

wandb.finish()
