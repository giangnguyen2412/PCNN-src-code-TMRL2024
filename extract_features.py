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
import random
import pdb
import faiss

from tqdm import tqdm
from torchvision import datasets, models, transforms
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torchvision
MODEL1 = torchvision.models.resnet34(pretrained=True).cuda()
MODEL1.eval()
feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor = nn.DataParallel(feature_extractor)
fc = MODEL1.fc

Dataset = Dataset()
RunningParams = RunningParams()

RETRIEVE_TOP1_NEAREST = True

in_features = MODEL1.fc.in_features

imagenet_dataset = ImageFolderWithPaths('/home/giang/Downloads/datasets/imagenet1k-val',
                                         Dataset.data_transforms['train'])

if RunningParams.DOGS_TRAINING == True:
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/datasets/Dogs_train', transform=Dataset.data_transforms['train'])
else:
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/train', transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if RETRIEVE_TOP1_NEAREST is True:
    INDEX_FILE = 'faiss/faiss_Dogs_class_idx_dict_RN34.npy'

    print("Building FAISS index........................................")
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

#####################################################################################

# Use RN34 to determine whether an image or correct or wrong
resnet34 = models.resnet34(pretrained=True).eval().cuda()
resnet34.eval()
resnet34 = nn.DataParallel(resnet34)

if RunningParams.DOGS_TRAINING is True:
    def load_imagenet_dog_label():
        dog_id_list = list()
        input_f = open("/home/giang/Downloads/ImageNet_Dogs_dataset/dog_type.txt")
        for line in input_f:
            dog_id = (line.split('-')[0])
            dog_id_list.append(dog_id)
        return dog_id_list

    dogs_id = load_imagenet_dog_label()


set = 'test'
data_dir = '/home/giang/Downloads/datasets/'

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                      Dataset.data_transforms['train'])
              for x in ['Dogs_{}'.format(set)]}

train_loader = torch.utils.data.DataLoader(
    image_datasets['Dogs_{}'.format(set)],
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,
    pin_memory=True,
)

dataset_sizes = {x: len(image_datasets[x]) for x in ['Dogs_{}'.format(set)]}

###########################################################################

depth_of_pred = 1
correct_cnt = 0
total_cnt = 0

faiss_nn_dict = dict()
nondog = 0
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    embeddings = torch.flatten(embeddings, start_dim=1)
    if RETRIEVE_TOP1_NEAREST is True:
        out = resnet34(data.cuda())
        embeddings = embeddings.cpu().detach().numpy()
        model1_p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(model1_p, depth_of_pred, dim=1)
        for sample_idx in range(data.shape[0]):
            base_name = os.path.basename(paths[sample_idx])
            gt_id = label[sample_idx].item()

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

                if depth_of_pred == 1:  # For val and test sets
                    _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]),
                                                    RunningParams.k_value)

                    for id in range(indices.shape[1]):
                        id = loader.dataset.indices[indices[0, id]]
                        nn_list.append(loader.dataset.dataset.imgs[id][0])
                    faiss_nn_dict[base_name] = nn_list
                else:

                    if i == 0:  # top-1 predictions --> Enrich top-1 prediction samples
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]),
                                                        faiss_index.ntotal)

                        for j in range(depth_of_pred):  # Make up x NN sets from top-1 predictions
                            nn_list = list()

                            if predicted_idx == gt_id:
                                key = 'Correct_{}_{}_'.format(i, j) + base_name
                                min_id = (j * RunningParams.k_value) + 1  # 3 NNs for one NN set
                                max_id = ((j * RunningParams.k_value) + RunningParams.k_value) + 1
                            else:
                                key = 'Wrong_{}_{}_'.format(i, j) + base_name
                                min_id = j * RunningParams.k_value  # 3 NNs for one NN set
                                max_id = (j * RunningParams.k_value) + RunningParams.k_value

                            for id in range(min_id, max_id):
                                id = loader.dataset.indices[indices[0, id]]
                                nn_list.append(loader.dataset.dataset.imgs[id][0])

                            faiss_nn_dict[key] = dict()
                            faiss_nn_dict[key]['NNs'] = nn_list
                            faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                            faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()

                    else:
                        if predicted_idx == gt_id:
                            key = 'Correct_{}_'.format(i) + base_name
                            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]),
                                                            RunningParams.k_value + 1)
                            indices = indices[:, 1:]  # skip the first NN
                        else:
                            key = 'Wrong_{}_'.format(i) + base_name
                            _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]),
                                                            RunningParams.k_value)

                        for id in range(indices.shape[1]):
                            id = loader.dataset.indices[indices[0, id]]
                            nn_list.append(loader.dataset.dataset.imgs[id][0])

                        faiss_nn_dict[key] = dict()
                        faiss_nn_dict[key]['NNs'] = nn_list
                        faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                        faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()

print("Non-dogs: {} files".format(nondog))
np.save('faiss/dogs/faiss_SDogs_{}_RN34_top1.npy'.format(set), faiss_nn_dict)
