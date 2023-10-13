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
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

Dataset = Dataset()
RunningParams = RunningParams()

depth_of_pred = 5
set = 'test'

MODEL1_RESNET = True
import torchvision
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
if MODEL1_RESNET is True:

    if RunningParams.resnet == 50:
        model = torchvision.models.resnet50(pretrained=True).cuda()
    elif RunningParams.resnet == 34:
        model = torchvision.models.resnet34(pretrained=True).cuda()
    elif RunningParams.resnet == 18:
        model = torchvision.models.resnet18(pretrained=True).cuda()
    model.fc = nn.Linear(model.fc.in_features, 196)

    my_model_state_dict = torch.load(
        '/home/giang/Downloads/advising_network/PyTorch-Stanford-Cars-Baselines/model_best_rn{}.pth.tar'.format(RunningParams.resnet), map_location=torch.device('cpu'))
    model.load_state_dict(my_model_state_dict['state_dict'], strict=True)
else:
    model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
    new_linear = nn.Linear(model.classifier[1].in_features, 196)
    new_classifier = nn.Sequential(nn.Dropout(0.2), new_linear)
    model.classifier = new_classifier

    my_model_state_dict = torch.load(
        '/home/giang/Downloads/advising_network/PyTorch-Stanford-Cars-Baselines/model_best_mobilenet_v2.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(my_model_state_dict['state_dict'], strict=True)

model.eval()

MODEL1 = model.cuda()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature

if MODEL1_RESNET is True:
    in_features = model.fc.in_features
else:
    in_features = 1280
    # Define the average pooling layer
    avgpool = nn.AdaptiveAvgPool2d(1)

    # Append the average pooling layer to the feature extractor
    feature_extractor.add_module("19", avgpool)

feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/Cars/Stanford-Cars-dataset/train',
                                     transform=data_transform)

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if MODEL1_RESNET is True:
    INDEX_FILE = 'faiss/cars/INDEX_file_adv_process_for_Cars_rn{}.npy'.format(RunningParams.resnet)
else:
    INDEX_FILE = 'faiss/cars/INDEX_file_adv_process_for_Cars_mobilenet_v2.npy'

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

# data_dir = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/{}'.format(set)
if set == 'train':
    data_dir = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/train'
elif set == 'test':
    data_dir = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/test'
else:
    exit(-1)

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(data_dir, data_transform)
train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=128,
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
    if len(train_loader.dataset.classes) < 196:
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
            faiss_nn_dict[base_name][key]['C_confidence'] = score[sample_idx][key].item()
            # breakpoint()

save_file = 'faiss/advising_process_{}_Cars.npy'.format(set)
print(save_file)
print(depth_of_pred)
np.save(save_file, faiss_nn_dict)
