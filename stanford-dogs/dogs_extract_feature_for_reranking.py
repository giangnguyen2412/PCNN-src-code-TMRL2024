# Extract NNs for advising process. Go into each top-predicted class,
# and extract the NNs for the input image in that class and put into a dictionary.
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
RunningParams = RunningParams('DOGS')

MODEL1_RESNET = True
depth_of_pred = 10
print(depth_of_pred)
set = 'test'

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

correct_cnt = 0
total_cnt = len(image_datasets['train'])

faiss_nn_dict = dict()
cnt = 0

for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
# for (data, label, paths), (data2, label2, paths2) in tqdm(zip(train_loader, std_train_loader)):
    if len(train_loader.dataset.classes) < 120:
        for sample_idx in range(data.shape[0]):
            tgt = label[sample_idx].item()
            class_name = train_loader.dataset.classes[tgt]
            id = faiss_dataset.class_to_idx[class_name]
            label[sample_idx] = id

    if MODEL1_RESNET is True:
        embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    else:
        embeddings = feature_extractor(data2.cuda())  # 512x1 for RN 18

    embeddings = torch.flatten(embeddings, start_dim=1)
    embeddings = embeddings.cpu().detach().numpy()

    if MODEL1_RESNET is True:
        out = MODEL1(data.cuda())
    else:
        _, out, _, _, _ = MODEL1(data.cuda())

    out = out.cpu().detach()
    model1_p = torch.nn.functional.softmax(out, dim=1)

    score_top1, index_top1 = torch.topk(model1_p, 1, dim=1)

    score, index = torch.topk(model1_p, depth_of_pred, dim=1)

    for sample_idx in range(data.shape[0]):
        base_name = os.path.basename(paths[sample_idx])
        gt_id = label[sample_idx]

        faiss_nn_dict[base_name] = dict()

        for i in range(depth_of_pred):
            # Get the top-k predicted label
            predicted_idx = index[sample_idx][i].item()

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
            faiss_nn_dict[base_name][key]['C_confidence'] = score[sample_idx][key]


print(cnt)
save_file = f'{RunningParams.prj_dir}/faiss/advising_process_top1_SDogs.npy'
print(save_file)
print(set)
print(depth_of_pred)
np.save(save_file, faiss_nn_dict)
