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
import faiss

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from torchvision import datasets, models, transforms
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


Dataset = Dataset()
RunningParams = RunningParams()


HIGHPERFORMANCE_FEATURE_EXTRACTOR = RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR
if HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
    from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    # Freeze backbone (for training only)
    for param in list(resnet.parameters())[:-2]:
        param.requires_grad = False
    # to CUDA
    inat_resnet = resnet.cuda()
    MODEL1 = inat_resnet
    MODEL1.eval()
else:
    import torchvision

    inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
    inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
    my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
    inat_resnet.load_state_dict(my_model_state_dict, strict=True)
    # Freeze backbone (for training only)
    for param in list(inat_resnet.parameters())[:-2]:
        param.requires_grad = False
    # to CUDA
    inat_resnet.cuda()
    MODEL1 = inat_resnet
    MODEL1.eval()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

in_features = 2048
print("Building FAISS index...! Training set is the knowledge base.")

faiss_dataset = datasets.ImageFolder(f'{RunningParams.parent_dir}/datasets/CUB/advnet/train',
                                     transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
    INDEX_FILE = 'faiss/cub/NeurIPS22_faiss_CUB200_class_idx_dict_HP_extractor.npy'
    print(INDEX_FILE)
else:
    exit(-1)

if os.path.exists(INDEX_FILE):
    print("FAISS class index exists!")
    faiss_nns_class_dict = np.load(INDEX_FILE, allow_pickle="False", ).item()
    targets = faiss_data_loader.dataset.targets
    faiss_data_loader_ids_dict = dict()
    faiss_loader_dict = dict()
    for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
        faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if targets[x] == class_id] # check this value
        class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
        class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
        faiss_loader_dict[class_id] = class_id_loader
else:
    print("FAISS class index NOT exists! Creating class index.........")
    targets = faiss_data_loader.dataset.targets
    faiss_data_loader_ids_dict = dict()
    faiss_nns_class_dict = dict()
    faiss_loader_dict = dict()
    for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
        faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if targets[x] == class_id]
        class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
        class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
        stack_embeddings = []
        for batch_idx, (data, label) in enumerate(class_id_loader):
            input_data = data.detach()
            embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
            embeddings = torch.flatten(embeddings, start_dim=1)

            stack_embeddings.append(embeddings.cpu().detach().numpy())
        stack_embeddings = np.concatenate(stack_embeddings, axis=0)
        descriptors = np.vstack(stack_embeddings)

        cpu_index = faiss.IndexFlatL2(in_features)
        # faiss_gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        #     cpu_index
        # )
        faiss_gpu_index = cpu_index

        faiss_gpu_index.add(descriptors)
        faiss_nns_class_dict[class_id] = faiss_gpu_index
        faiss_loader_dict[class_id] = class_id_loader
    np.save(INDEX_FILE, faiss_nns_class_dict)

from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
resnet.load_state_dict(my_model_state_dict, strict=True)
# Freeze backbone (for training only)
for param in list(resnet.parameters())[:-2]:
    param.requires_grad = False
# to CUDA
inat_resnet = resnet.cuda()
MODEL1 = inat_resnet
MODEL1.eval()

MODEL1 = nn.DataParallel(MODEL1).eval()

set = 'train'
# data_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/{}'.format(set)
data_dir = f'{RunningParams.parent_dir}/datasets/CUB/train1'  ##################################

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(data_dir, Dataset.data_transforms['train'])
train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=128,
    shuffle=True,  # Don't turn shuffle to False --> model works wrongly
    num_workers=16,
    pin_memory=True,
)

depth_of_pred = 10
correct_cnt = 0
total_cnt = 0

MODEL1.eval()

seed = 102
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

faiss_nn_dict = dict()
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    if batch_idx == 1:
        break
    if len(train_loader.dataset.classes) < 200:
        for sample_idx in range(data.shape[0]):
            tgt = label[sample_idx].item()
            class_name = train_loader.dataset.classes[tgt]
            id = faiss_dataset.class_to_idx[class_name]
            label[sample_idx] = id

    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    embeddings = embeddings.cpu().detach().numpy()

    out = MODEL1(data.cuda())
    model1_p = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.topk(model1_p, depth_of_pred, dim=1)
    for sample_idx in range(data.shape[0]):
        base_name = os.path.basename(paths[sample_idx])
        gt_id = label[sample_idx]

        # correct_cnt += torch.sum(index.squeeze() == label.cuda())
        # total_cnt += data.shape[0]

        key = paths[sample_idx]
        val = list()

        for i in range(depth_of_pred):
            # Get the top-k predicted label
            predicted_idx = index[sample_idx][i].item()

            # Dataloader and knowledge base upon the predicted class
            loader = faiss_loader_dict[predicted_idx]
            faiss_index = faiss_nns_class_dict[predicted_idx]
            nn_list = list()

            if depth_of_pred == 1:  # For val and test sets
                _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value)

                for id in range(indices.shape[1]):
                    id = loader.dataset.indices[indices[0, id]]
                    nn_list.append(loader.dataset.dataset.imgs[id][0])
                faiss_nn_dict[paths[sample_idx]] = nn_list[0]
            else:

                # key = base_name


                if i == 0:  # top-1 predictions --> Enrich top-1 prediction samples
                    _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), faiss_index.ntotal)

                    for j in range(depth_of_pred):  # Make up x NN sets from top-1 predictions
                        nn_list = list()

                        if predicted_idx == gt_id:
                            min_id = (j * RunningParams.k_value) + 1  # 3 NNs for one NN set
                            max_id = ((j * RunningParams.k_value) + RunningParams.k_value) + 1
                        else:
                            min_id = j * RunningParams.k_value  # 3 NNs for one NN set
                            max_id = (j * RunningParams.k_value) + RunningParams.k_value

                        for id in range(min_id, max_id):
                            id = loader.dataset.indices[indices[0, id]]
                            nn_list.append(loader.dataset.dataset.imgs[id][0])

                        val.append(nn_list[0])

                else:
                    if predicted_idx == gt_id:
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value+1)
                        indices = indices[:, 1:]  # skip the first NN
                    else:
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value)

                    for id in range(indices.shape[1]):
                        id = loader.dataset.indices[indices[0, id]]
                        nn_list.append(loader.dataset.dataset.imgs[id][0])

                    val.append(nn_list[0])

        faiss_nn_dict[key] = val


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# Your dictionary
image_dict = faiss_nn_dict.copy()

# Get the first key-value pair in the dictionary
first_key = list(image_dict.keys())[0]
first_value = image_dict[first_key]

# Get the parent directory name of the query image file
query_dir_name = os.path.basename(os.path.dirname(first_key))

# Load and resize the query image
query_img = Image.open(first_key)
query_img = query_img.resize((400, 400))

# Create a new figure with 3 rows
fig = plt.figure(figsize=(32, 10), dpi=200)

# Plot the query image in the first row
ax = plt.subplot(3, 1, 1)
plt.subplots_adjust(hspace=0.05)
ax.imshow(query_img)
ax.set_title(f'Query: {os.path.basename(os.path.dirname(first_value[0]))}', fontsize=18, color='green', weight='bold')
ax.axis('off')

# Plot the first 10 images in the second row
for i in range(10):
    img = Image.open(first_value[i])
    img_resized = img.resize((300, 300))
    img_resized = np.array(img_resized)
    ax = plt.subplot(3, 10, i+11)  # Starting from 11th position
    ax.imshow(img_resized)
    if i == 4:
        ax.set_title('Top-1: {}'.format(os.path.basename(os.path.dirname(first_value[i]))), fontsize=16, weight='bold')
    ax.axis('off')  # Hide axes

# Plot the last 9 images in the third row
for i in range(9):
    img = Image.open(first_value[i+10])
    img_resized = img.resize((300, 300))
    img_resized = np.array(img_resized)
    ax = plt.subplot(3, 10, i+22)  # Starting from 21st position
    ax.imshow(img_resized)
    ax.set_title('Top-{}: {}'.format(i+2, os.path.basename(os.path.dirname(first_value[i+10]))), fontsize=10)
    ax.axis('off')  # Hide axes

# Adjust the spacing between rows to minimize white spaces
plt.subplots_adjust(hspace=0.1)

# Save the figure as a PDF
plt.savefig("final_image.pdf", bbox_inches='tight', pad_inches=0.25)