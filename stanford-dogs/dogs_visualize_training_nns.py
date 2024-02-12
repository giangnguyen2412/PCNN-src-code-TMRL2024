# Visualize training NNs for AdvNet Cars-196

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Dataset = Dataset()
HelperFunctions = HelperFunctions()
RunningParams = RunningParams('DOGS')

MODEL1_RESNET = True
set = 'train'

torch.manual_seed(42)

################################################################################
import torchvision

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

MODEL1 = model.cuda()
MODEL1.eval()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

################################################################

if RunningParams.resnet == 34 or RunningParams.resnet == 18:
    in_features = 512
elif RunningParams.resnet == 50:
    in_features = 2048

print("Building FAISS index...! Training set is the knowledge base.")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

# faiss dataset contains images using as the knowledge based for KNN retrieval
faiss_dataset = datasets.ImageFolder(f'{RunningParams.parent_dir}/{RunningParams.train_path}',
                                     transform=data_transform)

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=64,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

INDEX_FILE = f'{RunningParams.prj_dir}/faiss/dogs/INDEX_file_adv_process_rn{RunningParams.resnet}.npy'
print(INDEX_FILE)

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

if MODEL1_RESNET is True:
    MODEL1 = nn.DataParallel(MODEL1).eval().cuda()

    data_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'

    image_datasets = dict()
    image_datasets['train'] = ImageFolderWithPaths(data_dir, Dataset.data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=128,
        shuffle=False,  # Don't turn shuffle to False --> model works wrongly
        num_workers=4,
        pin_memory=True,
    )

########################################################################

depth_of_pred = 10

if set == 'test':
    depth_of_pred = 1

correct_cnt = 0
total_cnt = 0

MODEL1.eval()

seed = 201
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

faiss_nn_dict = dict()
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    # if batch_idx == 1:
    #     break
    if len(train_loader.dataset.classes) < 120:
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

                if i == 0:  # top-1 predictions --> Enrich top-1 prediction samples
                    _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), faiss_index.ntotal)

                    if set == 'val':
                        width_of_pred = 1
                    else:
                        width_of_pred = depth_of_pred

                    for j in range(width_of_pred):  # Make up x NN sets from top-1 predictions
                        nn_list = list()

                        if predicted_idx == gt_id:
                            min_id = (j * RunningParams.k_value) + 1  # 3 NNs for one NN set
                            max_id = ((j * RunningParams.k_value) + RunningParams.k_value) + 1
                        else:
                            min_id = j * RunningParams.k_value  # 3 NNs for one NN set
                            max_id = (j * RunningParams.k_value) + RunningParams.k_value

                        for id in range(min_id, max_id):
                            # print(id)
                            # print(indices)
                            # print(loader.dataset.indices)
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

import random
random.seed(42)
# Shuffle the keys of the dictionary
keys = list(image_dict.keys())
random.shuffle(keys)

# Create a new dictionary with the shuffled keys
shuffled_dict = {key: image_dict[key] for key in keys}

breakpoint()

i = 0
for first_key, first_value in shuffled_dict.items():
    i += 1
    if i == 20:
        break
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
    ax.set_title(f'Query: {HelperFunctions.extract_dog_name(os.path.basename(os.path.dirname(first_value[0])))}', fontsize=18, color='green', weight='bold')
    ax.axis('off')

    # Plot the first 10 images in the second row
    for i in range(10):
        img = Image.open(first_value[i])
        img_resized = img.resize((300, 300))
        img_resized = np.array(img_resized)
        ax = plt.subplot(3, 10, i+11)  # Starting from 11th position
        ax.imshow(img_resized)
        if i == 4:
            ax.set_title('Top-1: {}'.format(HelperFunctions.extract_dog_name(os.path.basename(os.path.dirname(first_value[i])))), fontsize=16, weight='bold')
        ax.axis('off')  # Hide axes

    # Plot the last 9 images in the third row
    for i in range(9):
        img = Image.open(first_value[i+10])
        img_resized = img.resize((300, 300))
        img_resized = np.array(img_resized)
        ax = plt.subplot(3, 10, i+22)  # Starting from 21st position
        ax.imshow(img_resized)
        ax.set_title('Top-{}: {}'.format(i+2, HelperFunctions.extract_dog_name(os.path.basename(os.path.dirname(first_value[i+10])))), fontsize=8)
        ax.axis('off')  # Hide axes

    # Adjust the spacing between rows to minimize white spaces
    plt.subplots_adjust(hspace=0.1)

    # Save the figure as a PDF
    plt.savefig(f"final_image_{os.path.basename(first_key)}.pdf", bbox_inches='tight', pad_inches=0.25)
