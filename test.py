import numpy as np
import os
import random

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


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

train_data = ImageFolder(
    root=f'{RunningParams.parent_dir}/{RunningParams.train_path}', transform=data_transform
)

train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=12000,
            shuffle=False,
            num_workers=40,
            pin_memory=True,
        )

val_data = ImageFolder(
    # ImageNet train folder
    root=f'{RunningParams.parent_dir}/{RunningParams.test_path}', transform=data_transform
)


# filename = '/home/giang/Downloads/advising_network/faiss/advising_process_test_top1_HP_MODEL1_HP_FE.npy'
filename = '/home/giang/Downloads/advising_network/stanford-dogs/dogs_knn_algorithm_S_scores.npy'
print(filename)
all_similarity_scores = np.load(filename)
# kbc = np.load(filename, allow_pickle=True).item()

# breakpoint()

# Compute kNN accuracy, ignoring the padded samples
K_values = [20]
accuracies = []

from tqdm import tqdm

for K in K_values:
    avg_non_gt_scores = []
    avg_gt_scores = []
    correct_cnt = 0
    for i in tqdm(range(8580)):  # Only consider original test samples
        # Get the indices of the top K similar training samples
        top_k_indices = np.argsort(-all_similarity_scores[i])[:K]

        indices = np.argsort(-all_similarity_scores[i])
        non_gt_scores = [all_similarity_scores[i][idx] for idx in indices if train_data.targets[idx] != val_data.targets[i]]
        avg_non_gt_score = sum(non_gt_scores)/len(non_gt_scores)
        avg_non_gt_scores.append(avg_non_gt_score)

        gt_scores = [all_similarity_scores[i][idx] for idx in indices if train_data.targets[idx] == val_data.targets[i]]
        avg_gt_score = sum(gt_scores)/len(gt_scores)
        avg_gt_scores.append(avg_gt_score)

        # Get the labels of the top K similar training samples
        top_k_labels = [train_data.targets[idx] for idx in top_k_indices]
        # Predict the label based on the majority vote
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        # Check if the prediction is correct
        if predicted_label == val_data.targets[i]:
            correct_cnt += 1

    # Calculate and print the accuracy
    acc = 100 * correct_cnt / 8580  # Dividing by num_original_test_samples to ignore padded samples
    accuracies.append(acc)
    print(f"The accuracy of kNN at K = {K} is {acc}%")
    print("Gt scores: ", sum(avg_gt_scores)/len(avg_gt_scores))
    print("Non gt scores: ", sum(avg_non_gt_scores)/len(avg_non_gt_scores))

