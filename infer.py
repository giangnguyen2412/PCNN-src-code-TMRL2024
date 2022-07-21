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
import argparse
from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from models import MyCustomResnet18, AdvisingNetwork
from params import RunningParams
from helpers import HelperFunctions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_models/best_model_visionary-dust-173.pt',
                        help='Model check point')
    parser.add_argument('--eval_dataset', type=str,
                        default='/home/giang/Downloads/datasets/imagenet-sketch',
                        help='Evaluation dataset')

    args = parser.parse_args()
    model_path = args.ckpt
    print(args)

    model = AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    model.eval()

    RunningParams = RunningParams()

    data_dir = '/home/giang/Downloads/datasets/'
    val_datasets = ['imagenet-sketch']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), RunningParams.data_transforms[x])
                      for x in val_datasets}

    dataset_sizes = {x: len(image_datasets[x]) for x in val_datasets}

    model1_name = 'resnet18'
    MODEL1 = models.resnet18(pretrained=True).eval().cuda()

    for x in val_datasets:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=RunningParams.batch_size,
            shuffle=True,  # turn shuffle to True
            num_workers=8,
            pin_memory=True,
        )

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

            saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)
            explanation = torch.amax(saliency, dim=(1, 2, 3,))  # normalize the heatmaps
            explanation = torch.div(saliency, explanation.reshape(data.shape[0], 1, 1, 1))  # scale to 0->1

            # Incorporate explanations
            if RunningParams.XAI_method == 'No-XAI':
                inputs = x
            elif RunningParams.XAI_method == 'GradCAM':
                inputs = explanation * x  # multiply query & heatmap

            labels = model2_gt

            if RunningParams.advising_network is True:
                output = model(x, saliency, model1_p)
                p = torch.nn.functional.softmax(output, dim=1)
                _, preds = torch.max(p, 1)

            else:
                ds_output, fc_output = model(inputs)
                # print(fc_output[0][0])
                p = torch.nn.functional.softmax(ds_output, dim=1)
                _, preds = torch.max(p, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

            yes_cnt += sum(preds)
            true_cnt += sum(labels)

        epoch_acc = running_corrects.double() / len(image_datasets[x])
        yes_ratio = yes_cnt.double() / len(image_datasets[x])
        true_ratio = true_cnt.double() / len(image_datasets[x])

        print('{} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
            x, epoch_acc * 100, yes_ratio * 100, true_ratio * 100))


