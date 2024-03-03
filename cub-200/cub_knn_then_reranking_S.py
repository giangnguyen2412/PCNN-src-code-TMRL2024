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

from transformer import Transformer_AdvisingNetwork
reranker = Transformer_AdvisingNetwork()
reranker = nn.DataParallel(reranker).cuda()

model_path = 'best_models/best_model_decent-pyramid-3156.pt'
model_path = os.path.join(RunningParams.prj_dir, model_path)
checkpoint = torch.load(model_path)
running_params = checkpoint['running_params']
RunningParams.XAI_method = running_params.XAI_method

reranker.load_state_dict(checkpoint['model_state_dict'])
reranker.eval()


from torch.nn.functional import cosine_similarity

train_data = ImageFolder(
    # ImageNet train folder
    root=f"{RunningParams.parent_dir}/datasets/CUB/train1", transform=Dataset.data_transforms['train']
)

train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=256,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

val_data = ImageFolder(
    # ImageNet train folder
    root=f"{RunningParams.parent_dir}/datasets/CUB/test0", transform=Dataset.data_transforms['val']
)

test_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=256,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

import torch


def get_samples_and_targets_by_indices(indices, data_loader):
    samples = []
    targets = []

    for idx in indices:
        # Access the dataset directly using the index
        sample, target = data_loader.dataset[idx]
        samples.append(sample)
        targets.append(target)

    # Stack the list of tensors into a single tensor
    samples_tensor = torch.stack(samples)  # Creates a tensor of shape Nx3x224x224
    targets_tensor = torch.tensor(targets).unsqueeze(1)  # Creates a tensor of shape Nx1

    return samples_tensor, targets_tensor


N_test = len(val_data)

random.seed(42)
np.random.seed(42)

# Processing test_loader
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
all_val_concatted = all_val_concatted.reshape(-1, 2048)
Query = torch.from_numpy(all_val_concatted)
Query = Query.cuda()
Query = F.normalize(Query, dim=1)

# Initialize lists for train_loader embeddings and labels
all_train_embds = []
all_train_labels = []

# Existing code for processing train_loader and computing similarities
all_similarity_scores = []
target_labels = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        labels = HelperFunctions.to_np(target)

        embeddings = feature_extractor(data)
        embeddings = embeddings.view(-1, 2048)
        embeddings = F.normalize(embeddings, dim=1)
        q_results = torch.einsum("id,jd->ij", Query, embeddings).to("cpu")

        # Save embeddings and labels from train_loader
        all_train_embds.append(HelperFunctions.to_np(embeddings))
        all_train_labels.append(labels)

        all_similarity_scores.append(q_results)
        target_labels.append(target)

# Convert train_loader embeddings and labels to numpy arrays and concatenate
all_train_embds_concatted = HelperFunctions.concat(all_train_embds)
all_train_labels_concatted = HelperFunctions.concat(all_train_labels)

# Reshape and normalize the concatenated train_loader embeddings
Database = torch.from_numpy(all_train_embds_concatted)
Database = Database.cuda()
Database = F.normalize(Database, dim=1)

# Convert to numpy arrays for labels
labels_np = torch.cat(target_labels, -1)
val_labels_np = np.concatenate(all_val_labels)

all_similarity_scores = torch.cat(all_similarity_scores, 1)

K_values = [20, 50]
N_values = [50, 200]

scores = {}
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

# for K in K_values:
for K, N in zip(K_values, N_values):
    print(K, N)
    correct_cnt = 0
    reranked_scores = {}

    for i in tqdm(range(N_test)):

        # Get top N candidates for this query
        concat_ts = all_similarity_scores[i].cuda()
        sorted_ts = torch.argsort(concat_ts).cuda()
        sorted_topN = sorted_ts[-N:]

        query_features, query_targets = get_samples_and_targets_by_indices([i], test_loader)
        query_features = query_features.repeat(N, 1, 1, 1)
        nn_features, nn_targets = get_samples_and_targets_by_indices(sorted_topN, train_loader)
        nn_features = nn_features.unsqueeze(1)
        nn_targets = nn_targets.squeeze().cuda()
        # Assuming reranker takes a pair of features and returns a new similarity score
        # Adjust the following line based on the actual input format of your reranker
        new_scores, _, _, _ = reranker(query_features, nn_features, scores=None)
        new_scores = new_scores.squeeze()

        # Sort the new scores and select top-K
        _, reranked_topK_indices = torch.topk(new_scores, K)
        reranked_topK_targets = nn_targets[reranked_topK_indices]

        # Determine the most common label among the top-K
        predicted_label = reranked_topK_targets.mode()[0].item()

        # Check if the prediction is correct
        if predicted_label == query_targets[0].item():
            correct_cnt += 1

        # sorted_ids = torch.argsort(new_scores, descending=True)
        # # Store the new scores, sorted by score
        # reranked_scores[i] = sorted_topN[torch.argsort(new_scores, descending=True)]

    acc = 100 * correct_cnt / N_test

    print("The accuracy of reranking at K = {} and N = {} is {}".format(K, N, acc))