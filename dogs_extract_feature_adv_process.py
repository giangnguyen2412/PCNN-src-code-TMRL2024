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

depth_of_pred = 10
set = 'RN34_SDogs_val'

import torchvision
MODEL1 = torchvision.models.resnet34(pretrained=True).cuda()
MODEL1.eval()
feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor = nn.DataParallel(feature_extractor)
fc = MODEL1.fc

################################################################
imagenet_dataset = ImageFolderWithPaths('/home/giang/Downloads/datasets/imagenet1k-val',
                                         Dataset.data_transforms['train'])

in_features = 512
print("Building FAISS index...! Training set is the knowledge base.")

faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/RN34_SDogs_train',
                                     transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

INDEX_FILE = 'faiss/cub/INDEX_file_adv_process_for_Dogs.npy'
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


# Use RN34 to determine whether an image is correct or wrong
resnet34 = models.resnet34(pretrained=True).eval().cuda()
resnet34.eval()
resnet34 = nn.DataParallel(resnet34)
MODEL1 = resnet34

if RunningParams.DOGS_TRAINING is True:
    def load_imagenet_dog_label():
        dog_id_list = list()
        input_f = open("/home/giang/Downloads/ImageNet_Dogs_dataset/dog_type.txt")
        for line in input_f:
            dog_id = (line.split('-')[0])
            dog_id_list.append(dog_id)
        return dog_id_list

    dogs_id = load_imagenet_dog_label()

########################################################################

data_dir = '/home/giang/Downloads/'

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                      Dataset.data_transforms['train'])
              for x in [set]}

train_loader = torch.utils.data.DataLoader(
    image_datasets[set],
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,
    pin_memory=True,
)

correct_cnt = 0
total_cnt = 0
nondog = 0

MODEL1.eval()

faiss_nn_dict = dict()

for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
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

        faiss_nn_dict[base_name] = dict()

        for i in range(depth_of_pred):
            # Get the top-k predicted label
            predicted_idx = index[sample_idx][i].item()

            if RunningParams.DOGS_TRAINING is True:
                # If the predicted ID is not a DOG ... then we randomize it to a Dog wnid
                dog_wnid = imagenet_dataset.classes[predicted_idx]
                if dog_wnid not in dogs_id:
                    dog_wnid = random.choice([wnid for wnid in dogs_id if wnid != dog_wnid])
                    nondog += 1

                # convert to Dog id
                predicted_idx = faiss_dataset.class_to_idx[dog_wnid]

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

save_file = 'faiss/advising_process_{}_top1_HP_MODEL1_HP_FE.npy'.format(set)
print(save_file)
print(depth_of_pred)
np.save(save_file, faiss_nn_dict)
