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

from transformer import Transformer_AdvisingNetwork
model = Transformer_AdvisingNetwork()
model = nn.DataParallel(model).cuda()

model_path = f'{RunningParams.prj_dir}/best_models/best_model_dainty-blaze-3325.pt'
checkpoint = torch.load(model_path)
running_params = checkpoint['running_params']

model.load_state_dict(checkpoint['model_state_dict'])
# # Define the average pooling layer
# avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
# # Append the average pooling layer to conv_layers
# model.module.conv_layers.add_module('8', avg_pool)
model.eval()

def compute_similarity(transformer_model, query_images, reference_images):
    # Assuming your model takes a batch of image pairs and returns a batch of similarity scores
    # Adjust this part based on your model's input and output format
    scores, _, _, _ = transformer_model(query_images, reference_images, None)
    return scores

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

print(RunningParams.train_path)
print(RunningParams.test_path)

N_test = len(val_data)

# Number of original test samples
num_original_test_samples = len(val_data)

# Create a DataLoader for the padded test dataset
padded_test_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=60,  # Keep the batch size consistent with your original setup
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# Initialize the array to store similarity scores with the new size
all_similarity_scores = np.zeros((8580, len(train_data)))  # Use 8580 here for the padded size
num_test_samples = 30

def compute_similarity_chunked(transformer_model, query_images, reference_images):
    """
    This function computes similarity scores between each query image and all reference images.
    It processes the reference images in chunks to avoid memory issues.
    """
    scores = []
    for query_img in query_images:
        query_scores = []
        for ref_chunk in torch.split(reference_images, 12000):  # Adjust chunk size as needed
            chunk_scores, _, _, _ = transformer_model(query_img.unsqueeze(0).expand_as(ref_chunk), ref_chunk, None)
            # breakpoint()
            query_scores.append(chunk_scores)
        scores.append(torch.cat(query_scores))
    return torch.stack(scores)

with torch.no_grad():
    all_similarity_scores = []  # This will be a list of tensors initially

    for batch_idx, (test_data, test_targets) in enumerate(tqdm(padded_test_loader)):
        # if batch_idx * padded_test_loader.batch_size > num_test_samples:
        #     print(batch_idx * padded_test_loader.batch_size)
        #     break  # Stop after reaching the desired number of test samples

        test_data = test_data.cuda()

        # Initialize a tensor to hold similarity scores for the current batch of test images
        batch_scores = torch.zeros((test_data.size(0), len(train_data)))

        for ref_batch_idx, (ref_data, ref_targets) in enumerate(tqdm(train_loader)):
            ref_data = ref_data.cuda()

            # Compute similarity scores for the current batch of test images against the current batch of reference images
            similarity_scores = compute_similarity_chunked(model, test_data, ref_data)
            p_scores = torch.sigmoid(similarity_scores)
            # breakpoint()

            # Store the similarity scores in the appropriate slice of the batch_scores tensor
            start_idx = ref_batch_idx * train_loader.batch_size
            end_idx = start_idx + ref_data.size(0)
            batch_scores[:, start_idx:end_idx] = p_scores.squeeze()

        # Add the batch scores to the all_similarity_scores list
        all_similarity_scores.append(batch_scores.cpu())

    # Concatenate all batch scores to form the final similarity score matrix
    all_similarity_scores = torch.cat(all_similarity_scores, dim=0)

# Convert all_similarity_scores to a NumPy array if it's not already one
all_similarity_scores_np = all_similarity_scores.numpy()

# Save the NumPy array to disk
np.save('dogs_knn_algorithm_S_scores.npy', all_similarity_scores_np)

print("Saved all_similarity_scores to disk as 'dogs_knn_algorithm_S_scores.npy'")

# Compute kNN accuracy, ignoring the padded samples
K_values = [5, 10, 20, 50, 100, 200]
accuracies = []

for K in K_values:
    correct_cnt = 0
    for i in range(num_original_test_samples):  # Only consider original test samples
        # Get the indices of the top K similar training samples
        top_k_indices = np.argsort(-all_similarity_scores[i])[:K]
        # Get the labels of the top K similar training samples
        top_k_labels = [train_data.targets[idx] for idx in top_k_indices]
        # Predict the label based on the majority vote
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        # Check if the prediction is correct
        if predicted_label == val_data.targets[i]:
            correct_cnt += 1

    # Calculate and print the accuracy
    acc = 100 * correct_cnt / num_original_test_samples  # Dividing by num_original_test_samples to ignore padded samples
    accuracies.append(acc)
    print(f"The accuracy of kNN at K = {K} is {acc}%")




