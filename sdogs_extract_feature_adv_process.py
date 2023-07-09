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

from tqdm import tqdm
from torchvision import datasets, models, transforms
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, StanfordDogsDataset

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

Dataset = Dataset()
RunningParams = RunningParams()

depth_of_pred = 5
set = 'test'

import torchvision.transforms as T
import math

def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return T.Compose([
        T.Resize((height, width)),
        T.Pad(pad_values),
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)


model = models.resnet34(pretrained=True)

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

from collections import OrderedDict
new_ckpt = OrderedDict()
ckpt = torch.load('/home/giang/Downloads/advising_network/stanford-dogs/HAL9002_RN34.pt')
print(ckpt['val_acc']/100)

for k, v in ckpt['model_state_dict'].items():
    new_k = k.replace('module.', '')
    new_ckpt[new_k] = v

model.load_state_dict(new_ckpt)

MODEL1 = model.cuda()
MODEL1.eval()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

in_features = 512

# Because it has a unique type of mapping the labels, we need reference_dataset to convert the predicted ids (StanfordDogsDataset) to ImageFolder ids
reference_dataset = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)

faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/advising_network/stanford-dogs/data/images/train',
                                     transform=preprocess)

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

INDEX_FILE = 'faiss/cub/INDEX_file_adv_process_for_SDogs.npy'
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

MODEL1 = nn.DataParallel(MODEL1).eval()

image_datasets = dict()
# image_datasets['train'] = StanfordDogsDataset(
#     root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', 'data'), set_type=set, transform=preprocess)

image_datasets['train'] = ImageFolderWithPaths('/home/giang/Downloads/advising_network/stanford-dogs/data/images/{}'.format(set),
                                     transform=preprocess)

train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=32,
    shuffle=False,  # Don't turn shuffle to False --> model works wrongly
    num_workers=16,
    pin_memory=True,
)

correct_cnt = 0
total_cnt = 0
nondog = 0

MODEL1.eval()

faiss_nn_dict = dict()

for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
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

        faiss_nn_dict[base_name] = dict()

        for i in range(depth_of_pred):
            # Get the top-k predicted label
            predicted_idx = index[sample_idx][i].item()

            ################################ CONVERT THE PREDICTED IDS #################################
            dog_name = reference_dataset.mapping[predicted_idx]
            predicted_idx = train_loader.dataset.class_to_idx[dog_name]
            ############################################################################################

            # Dataloader and knowledge base upon the predicted class
            loader = faiss_loader_dict[predicted_idx]
            faiss_index = faiss_nns_class_dict[predicted_idx]
            nn_list = list()

            key = i
            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), 6)

            for id in range(indices.shape[1]):
                id = loader.dataset.indices[indices[0, id]]
                nn_list.append(loader.dataset.dataset.imgs[id][0])

            faiss_nn_dict[base_name][key] = dict()
            faiss_nn_dict[base_name][key]['NNs'] = nn_list
            faiss_nn_dict[base_name][key]['Label'] = predicted_idx

save_file = 'faiss/advising_process_{}_SDogs.npy'.format(set)
print(save_file)
print(depth_of_pred)
np.save(save_file, faiss_nn_dict)
