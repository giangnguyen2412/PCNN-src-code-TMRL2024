# Running kNN classifier for Cars-196
import os
import random
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from datasets import Dataset
from params import RunningParams
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from helpers import HelperFunctions


RunningParams = RunningParams('DOGS')

HelperFunctions = HelperFunctions()
Dataset = Dataset()

import torchvision

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
ORIGINAL_FE = True
if ORIGINAL_FE is True:

    if RunningParams.resnet == 50:
        model = torchvision.models.resnet50(pretrained=True).cuda()
        model.fc = nn.Linear(2048, 120)
    elif RunningParams.resnet == 34:
        model = torchvision.models.resnet34(pretrained=True).cuda()
        model.fc = nn.Linear(512, 120)
    elif RunningParams.resnet == 18:
        model = torchvision.models.resnet18(pretrained=True).cuda()
        model.fc = nn.Linear(512, 120)

    print(f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth')
    my_model_state_dict = torch.load(
        f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth',
        map_location='cuda'
    )
    new_state_dict = {k.replace("model.", ""): v for k, v in my_model_state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)

    MODEL1 = model
    MODEL1.eval()

    feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
    feature_extractor.cuda()
    feature_extractor = nn.DataParallel(feature_extractor)
else:  # kNN-AdvNet
    from transformer import Transformer_AdvisingNetwork
    model = Transformer_AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    # model_path = f'{RunningParams.prj_dir}/best_models/best_model_copper-moon-3322.pt' # RN18
    # model_path = f'{RunningParams.prj_dir}/best_models/best_model_woven-deluge-3324.pt'  # RN34
    model_path = f'{RunningParams.prj_dir}/best_models/best_model_dainty-blaze-3325.pt'
    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']

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
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

val_data = ImageFolder(
    # ImageNet train folder
    root=f'{RunningParams.parent_dir}/{RunningParams.test_path}', transform=data_transform
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
    torch.manual_seed(42)

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

    saved_results = []
    target_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # if batch_idx == 2:
            #     break
            data = data.cuda()
            labels = HelperFunctions.to_np(target)

            embeddings = feature_extractor(data)
            embeddings = embeddings.view(-1, 2048)
            embeddings = F.normalize(embeddings, dim=1)
            q_results = torch.einsum("id,jd->ij", Query, embeddings).to("cpu")

            saved_results.append(q_results)
            target_labels.append(target)

    # Convert to numpy arrays
    labels_np = torch.cat(target_labels, -1)
    val_labels_np = np.concatenate(all_val_labels)

    saved_results = torch.cat(saved_results, 1)

    # Convert all_similarity_scores to a NumPy array if it's not already one
    all_similarity_scores_np = saved_results.numpy()

    # Save the NumPy array to disk
    np.save('dogs_knn_algorithm_scores.npy', all_similarity_scores_np)

    # Compute the top-1 accuracy of KNNs, save the KNN dictionary
    scores = {}
    import torch

    # K_values = [5, 10, 20, 50, 100, 200]
    K_values = [5]

    for K in K_values:
        # print("K = {}".format(K))
        correct_cnt = 0
        for i in tqdm(range(N_test)):
            concat_ts = saved_results[i].cuda()
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