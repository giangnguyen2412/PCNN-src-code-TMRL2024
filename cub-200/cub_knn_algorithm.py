import collections
import json
import os
import random
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from datasets import Dataset
from params import RunningParams
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from helpers import HelperFunctions


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

RunningParams = RunningParams()
HelperFunctions = HelperFunctions()
Dataset = Dataset()

ORIGINAL_FE = True
if ORIGINAL_FE is True:
    from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')

    resnet.load_state_dict(my_model_state_dict, strict=True)
    MODEL1 = resnet.cuda()

    # MODEL1 = torchvision.models.resnet50(pretrained=True)
    MODEL1.eval()

    feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
    feature_extractor.cuda()
    feature_extractor = nn.DataParallel(feature_extractor)
else:
    from transformer import Transformer_AdvisingNetwork
    model = Transformer_AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    # model_path = 'best_models/best_model_genial-plasma-3125.pt'
    model_path = 'best_models/best_model_decent-pyramid-3156.pt'
    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    model.load_state_dict(checkpoint['model_state_dict'])
    # Define the average pooling layer
    avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    # Append the average pooling layer to conv_layers
    model.module.conv_layers.add_module('8', avg_pool)

    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    feature_extractor = model.module.conv_layers
    feature_extractor.cuda()
    feature_extractor = nn.DataParallel(feature_extractor)

from torch.nn.functional import cosine_similarity

train_data = ImageFolder(
    # ImageNet train folder
    root=f"{RunningParams.parent_dir}/datasets/CUB/train1", transform=Dataset.data_transforms['train']
)

train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

val_data = ImageFolder(
    # ImageNet train folder
    root=f"{RunningParams.parent_dir}/datasets/CUB/test0", transform=Dataset.data_transforms['val']
)

test_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

N_test = len(val_data)

if True:
    random.seed(42)
    np.random.seed(42)

    all_val_embds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data = data.cuda()

            embeddings = HelperFunctions.to_np(feature_extractor(data))
            labels = HelperFunctions.to_np(target)

            all_val_embds.append(embeddings)
            all_val_labels.append(labels)

    all_val_concatted = HelperFunctions.concat(all_val_embds)
    all_val_labels_concatted = HelperFunctions.concat(all_val_labels)

    #
    all_val_concatted = all_val_concatted.reshape(-1, 2048)

    Query = torch.from_numpy(all_val_concatted)
    Query = Query.cuda()
    Query = F.normalize(Query, dim=1)

    all_similarity_scores = []
    target_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # if batch_idx == 2:
            #     break
            data = data.cuda()
            labels = HelperFunctions.to_np(target)

            embeddings = feature_extractor(data)
            embeddings = embeddings.view(-1, 2048)  # Using conv4 with 100352-sized embeddings only scores 78.03% at K=20
            embeddings = F.normalize(embeddings, dim=1)  # Same results if using F.cosine_similarity
            q_results = torch.einsum("id,jd->ij", Query, embeddings).to("cpu")

            all_similarity_scores.append(q_results)
            target_labels.append(target)

    # Convert to numpy arrays
    labels_np = torch.cat(target_labels, -1)
    val_labels_np = np.concatenate(all_val_labels)

    all_similarity_scores = torch.cat(all_similarity_scores, 1)

    # Compute the top-1 accuracy of KNNs, save the KNN dictionary
    scores = {}

    # Convert all_similarity_scores to a NumPy array if it's not already one
    all_similarity_scores_np = all_similarity_scores.numpy()

    # Save the NumPy array to disk
    np.save('cub_knn_algorithm_sim.npy', all_similarity_scores_np)
    np.save('cub_knn_algorithm_val_labels.npy', val_labels_np)

    print("Saved all_similarity_scores to disk as 'cub_knn_algorithm.npy'")

    K_values = [5, 10, 20, 50, 100, 200]
    # K_values = [5, 10]

    for K in K_values:
        # print("K = {}".format(K))
        correct_cnt = 0
        for i in tqdm(range(N_test)):
            concat_ts = all_similarity_scores[i].cuda()
            sorted_ts = torch.argsort(concat_ts).cuda()
            sorted_topk = sorted_ts[-K:]
            scores[i] = torch.flip(
                sorted_topk, dims=[0]
            )  # Move the closest to the head

            gt_id = val_labels_np[i]
            labels_np = labels_np.cuda()
            prediction = torch.argmax(torch.bincount(labels_np[scores[i]]))

            if prediction == gt_id:
                correctness = True
            else:
                correctness = False

            if correctness:
                correct_cnt += 1

        acc = 100 * correct_cnt / N_test

        print("The accuracy of kNN at K = {} is {}".format(K, acc))