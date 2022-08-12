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
from models import MyCustomResnet18, AdvisingNetwork, AdvisingNetworkv1
from params import RunningParams
from datasets import Dataset
from torchvision.datasets import ImageFolder
from visualize import Visualization
from helpers import HelperFunctions


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        data = original_tuple[0]
        label = original_tuple[1]
        if data.shape[0] == 1:
            print('gray images')
            data = torch.cat([data, data, data], dim=0)

        # make a new tuple that includes original and the path
        tuple_with_path = (data, label, path)
        return tuple_with_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_models/best_model_valiant-plasma-176.pt',
                        help='Model check point')
    parser.add_argument('--eval_dataset', type=str,
                        default='/home/giang/Downloads/datasets/imagenet1k-val',
                        help='Evaluation dataset')

    args = parser.parse_args()
    model_path = args.ckpt
    print(args)

    model = AdvisingNetworkv1()
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
    HelperFunctions = HelperFunctions()
    print(RunningParams.__dict__)

    data_dir = '/home/giang/Downloads/datasets/'
    # TODO: Using mask to make the predictions compatible with ImageNet-R, ObjectNet, ImageNet-A
    val_datasets = Dataset.test_datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), Dataset.data_transforms['val'])
    #                   for x in val_datasets}
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), Dataset.data_transforms['val'])
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

        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']
        confidence_dist = dict()
        for cat in categories:
            confidence_dist[cat] = list()

        gaussian_blur = transforms.CenterCrop(size=180)

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data.cuda()
            # x = gaussian_blur(x)
            ART_TESTS1 = True
            if ART_TESTS1 is False:
                x = torch.rand([16, 3, 224, 224]).cuda()
            gts = gt.cuda()

            # Step 1: Forward pass input x through MODEL1 - Explainer
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            score, index = torch.topk(model1_p, 1, dim=1)
            _, model1_ranks = torch.topk(model1_p, 1000, dim=1)
            predicted_ids = index.squeeze()

            # MODEL1 Y/N label for input x
            # model2_gt = (predicted_ids == gts) * 1  # 0 and 1

            # Generate heatmap explanation using MODEL1 and its predicted ids
            saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)
            # labels = model2_gt

            if RunningParams.advising_network is True:
                # Forward input, explanations, and softmax scores through MODEL2

                if ART_TESTS1:
                    # confidence ranges from 0 -> 100%
                    for i in range(0, 105, 1):
                        conf = i/100
                        other_conf = 1 - conf
                        other_vals = other_conf/999  # the confidence scores for other classes
                        model1_p = torch.full(model1_p.shape, other_vals)
                        for j in range(x.shape[0]):
                            idx = index[j].item()
                            model1_p[j][idx] = conf

                        output = model(x, saliency, model1_p)
                        model2_p = torch.nn.functional.softmax(output, dim=1)
                        score, index = torch.topk(model2_p, 1, dim=1)
                        if index[0].item() == 0:
                            print("MODEL2 says NO at Confidence: {}% ".format(i))
                        else:
                            print("MODEL2 says YES at Confidence: {}% ".format(i))
                exit(-1)
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

            if RunningParams.MODEL2_ADVISING is False and RunningParams.M2_VISUALIZATION is True:
                results = (preds == labels)
                for sample_idx in range(x.shape[0]):
                    result = results[sample_idx].item()
                    if result is True:
                        correctness = 'Correctly'
                    else:
                        correctness = 'Incorrectly'

                    label = preds[sample_idx].item()
                    if label == 1:
                        action = 'Accept'
                    else:
                        action = 'Reject'

                    model2_decision = correctness + action
                    query = pths[sample_idx]

                    # TODO: move this out to remove redundancy
                    save_dir = '/home/giang/Downloads/advising_net_training/vis/'
                    base_name = os.path.basename(query)
                    HelperFunctions.check_and_mkdir(os.path.join(save_dir, model2_decision))
                    save_path = os.path.join(save_dir, model2_decision, base_name)

                    gt_label = HelperFunctions.label_map.get(gts[sample_idx].item()).split(",")[0]
                    gt_label = gt_label[0].lower() + gt_label[1:]

                    pred_label = HelperFunctions.label_map.get(predicted_ids[sample_idx].item()).split(",")[0]
                    pred_label = pred_label[0].lower() + pred_label[1:]

                    confidence = int(score[sample_idx].item()*100)
                    confidence_dist[model2_decision].append(confidence)

                    # Visualization.visualize_model2_decisions(query,
                    #                                          gt_label,
                    #                                          pred_label,
                    #                                          model2_decision,
                    #                                          save_path,
                    #                                          save_dir,
                    #                                          confidence)

            yes_cnt += sum(preds)
            true_cnt += sum(labels)

        if RunningParams.M2_VISUALIZATION is True:
            for cat in categories:
                img_ratio = len(confidence_dist[cat])*100/dataset_sizes[ds]
                title = '{}: {:.2f}'.format(cat, img_ratio)
                Visualization.visualize_histogram_from_list(data=confidence_dist[cat],
                                                            title=title,
                                                            x_label='Confidence',
                                                            y_label='Images',
                                                            file_name=os.path.join(save_dir, cat + '.pdf'),
                                                            )

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





