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
from datasets import Dataset
from helpers import HelperFunctions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_models/best_model_efficient-aardvark-180.pt',
                        help='Model check point')
    parser.add_argument('--eval_dataset', type=str,
                        default='/home/giang/Downloads/datasets/imagenet-r',
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
    Dataset = Dataset()
    print(RunningParams.__dict__)

    data_dir = '/home/giang/Downloads/datasets/'
    # TODO: Using mask to make the predictions compatible with ImageNet-R, ObjectNet, ImageNet-A
    val_datasets = Dataset.test_datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), Dataset.data_transforms['val'])
                      for x in val_datasets}

    dataset_sizes = {x: len(image_datasets[x]) for x in val_datasets}

    model1_name = 'resnet18'
    MODEL1 = models.resnet18(pretrained=True).eval().cuda()

    for ds in val_datasets:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=RunningParams.batch_size,
            shuffle=True,  # turn shuffle to True
            num_workers=8,
            pin_memory=True,
        )

        running_corrects = 0
        advising_crt_cnt = 0
        yes_cnt = 0
        true_cnt = 0

        for batch_idx, (data, gt) in enumerate(tqdm(data_loader)):
            x = data.cuda()
            gts = gt.cuda()

            # Step 1: Forward pass input x through MODEL1 - Explainer
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            score, index = torch.topk(model1_p, 1, dim=1)
            _, model1_ranks = torch.topk(model1_p, 1000, dim=1)
            predicted_ids = index.squeeze()

            # MODEL1 Y/N label for input x
            model2_gt = (predicted_ids == gts) * 1  # 0 and 1

            # Generate heatmap explanation using MODEL1 and its predicted ids
            saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)
            labels = model2_gt

            if RunningParams.advising_network is True:
                # Forward input, explanations, and softmax scores through MODEL2
                output = model(x, saliency, model1_p)
                p = torch.nn.functional.softmax(output, dim=1)
                _, preds = torch.max(p, 1)

            # Running ADVISING process
            # TODO: Reduce compute overhead by only running on disagreed samples
            advising_steps = RunningParams.advising_steps
            # When using advising & still disagree & examining only top $advising_steps + 1
            while RunningParams.MODEL2_ADVISING is True and sum(preds) > 0:
                for k in range(1, RunningParams.advising_steps):
                    # top-k predicted labels
                    model1_topk = model1_ranks[:, k]
                    for pred_idx in range(x.shape[0]):
                        if preds[pred_idx] == 0:
                            # Change the predicted id to top-k if MODEL2 disagrees
                            predicted_ids[pred_idx] = model1_topk[pred_idx]
                    # Unsqueeze to fit into grad_cam()
                    index = torch.unsqueeze(predicted_ids, dim=1)
                    # Generate heatmap explanation using new category ids
                    advising_saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)

                    # TODO: If the explanation is No-XAI, the inputs in advising process don't change
                    if RunningParams.advising_network is True:
                        advising_output = model(x, advising_saliency, model1_p)
                        advising_p = torch.nn.functional.softmax(advising_output, dim=1)
                        _, preds = torch.max(advising_p, 1)
                break

            # If MODEL2 still disagrees, we revert the ids to top-1 predicted labels
            model1_top1 = model1_ranks[:, 0]
            for pred_idx in range(x.shape[0]):
                if preds[pred_idx] == 0:
                    # Change the predicted id to top-k if MODEL2 disagrees
                    predicted_ids[pred_idx] = model1_top1[pred_idx]

            # statistics
            if RunningParams.MODEL2_ADVISING:
                # Re-compute the accuracy of imagenet classification
                advising_crt_cnt += torch.sum(predicted_ids == gts)
                advising_labels = (predicted_ids == gts) * 1
                running_corrects += torch.sum(preds == advising_labels.data)
            else:
                running_corrects += torch.sum(preds == labels.data)

            yes_cnt += sum(preds)
            true_cnt += sum(labels)

        epoch_acc = running_corrects.double() / len(image_datasets[ds])
        yes_ratio = yes_cnt.double() / len(image_datasets[ds])
        true_ratio = true_cnt.double() / len(image_datasets[ds])

        if RunningParams.MODEL2_ADVISING is True:
            advising_acc = advising_crt_cnt.double() / len(image_datasets[ds])

            print('{} - Acc: {:.2f} - Yes Ratio: {:.2f} - Orig. accuracy: {:.2f} - Advising. accuracy: {:.2f}'.format(
                ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100, advising_acc * 100))
        else:
            print('{} - Acc: {:.2f} - Yes Ratio: {:.2f} - Orig. accuracy: {:.2f}'.format(
                ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100))





