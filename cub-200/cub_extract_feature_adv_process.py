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
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

Dataset = Dataset()
RunningParams = RunningParams()

MODEL1_RN50 = True
depth_of_pred = 10
set = 'test'

torch.manual_seed(42)

################################################################################
from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    f'{RunningParams.prj_dir}/pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
resnet.load_state_dict(my_model_state_dict, strict=True)
# Freeze backbone (for training only)
for param in list(resnet.parameters())[:-2]:
    param.requires_grad = False
# to CUDA
inat_resnet = resnet.cuda()
MODEL1 = inat_resnet
MODEL1.eval()

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

################################################################

in_features = 2048
print("Building FAISS index...! Training set is the knowledge base.")

faiss_dataset = datasets.ImageFolder(f'{RunningParams.parent_dir}/datasets/CUB/train1',
                                     transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=64,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

if MODEL1_RN50 is True:
    INDEX_FILE = f'{RunningParams.prj_dir}/faiss/cub/INDEX_file_adv_process.npy'
else:
    INDEX_FILE = f'{RunningParams.prj_dir}/faiss/cub/INDEX_file_adv_process_NTSNET.npy'

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

if MODEL1_RN50 is True:
    ################################################################
    from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        f'{RunningParams.prj_dir}/pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    # Freeze backbone (for training only)
    for param in list(resnet.parameters())[:-2]:
        param.requires_grad = False
    # to CUDA
    inat_resnet = resnet.cuda()
    MODEL1 = inat_resnet
    MODEL1.eval()

    MODEL1 = nn.DataParallel(MODEL1).eval()

    # data_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/{}'.format(set)
    data_dir = f'{RunningParams.parent_dir}/datasets/CUB/test0'

    image_datasets = dict()
    image_datasets['train'] = ImageFolderWithPaths(data_dir, Dataset.data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=128,
        shuffle=False,  # Don't turn shuffle to False --> model works wrongly
        num_workers=16,
        pin_memory=True,
    )
    print('Running MODEL1 being RN50!!!')

else:
    ################################################################
    import os
    from torch.autograd import Variable
    import torch.utils.data
    from torch.nn import DataParallel
    from core import model, dataset
    from torch import nn
    from datasets import Dataset
    from torchvision.datasets import ImageFolder
    from tqdm import tqdm
    net = model.attention_net(topN=6)
    ckpt = torch.load(f'{RunningParams.parent_dir}/NTS-Net/model.ckpt')

    net.load_state_dict(ckpt['net_state_dict'])

    # feature_extractor = nn.Sequential(*list(net.children())[:-1])  # avgpool feature
    # print(net)

    net.eval()
    net = net.cuda()
    net = DataParallel(net)
    MODEL1 = net

    Dataset = Dataset()

    from torchvision import transforms
    from PIL import Image

    data_transforms = transforms.Compose([
        transforms.Resize((600, 600), interpolation=Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # data_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/{}'.format(set)
    data_dir = f'{RunningParams.parent_dir}/datasets/CUB/test0'
    val_data = ImageFolderWithPaths(
        # ImageNet train folder
        root=data_dir, transform=data_transforms
    )

    std_val_data = ImageFolderWithPaths(
        # ImageNet train folder
        root=data_dir, transform=Dataset.data_transforms['val']
    )

    train_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    std_train_loader = torch.utils.data.DataLoader(
        std_val_data,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    print('Running MODEL1 being NTS-NET!!!')


########################################################################

correct_cnt = 0
total_cnt = 5794

MODEL1.eval()

faiss_nn_dict = dict()
cnt = 0

# Number of buckets
M = 20
bucket_limits = torch.linspace(0, 1, M + 1)
bucket_data = {'accuracies': torch.zeros(M), 'confidences': torch.zeros(M), 'counts': torch.zeros(M)}

for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
# for (data, label, paths), (data2, label2, paths2) in tqdm(zip(train_loader, std_train_loader)):
    if len(train_loader.dataset.classes) < 200:
        for sample_idx in range(data.shape[0]):
            tgt = label[sample_idx].item()
            class_name = train_loader.dataset.classes[tgt]
            id = faiss_dataset.class_to_idx[class_name]
            label[sample_idx] = id

    if MODEL1_RN50 is True:
        embeddings = feature_extractor(data.cuda())  # 512x1 for RN 18
    else:
        embeddings = feature_extractor(data2.cuda())  # 512x1 for RN 18

    embeddings = torch.flatten(embeddings, start_dim=1)
    embeddings = embeddings.cpu().detach().numpy()

    if MODEL1_RN50 is True:
        out = MODEL1(data.cuda())
    else:
        _, out, _, _, _ = MODEL1(data.cuda())

    model1_p = torch.nn.functional.softmax(out, dim=1)

    score_top1, index_top1 = torch.topk(model1_p, 1, dim=1)

#     # For ECE calculation: Get the maximum predicted probability and its corresponding label
#     max_probs, preds = torch.max(model1_p, 1)
#     correct = preds.eq(label.cuda()).float()
#
#     max_probs = max_probs.detach().cpu()
#     preds = preds.detach().cpu()
#     correct = correct.detach().cpu()
#
#     # Sort the confidences into buckets
#     for i in range(M):
#         in_bucket = max_probs.gt(bucket_limits[i]) & max_probs.le(bucket_limits[i + 1])
#         bucket_data['counts'][i] += in_bucket.float().sum()
#         bucket_data['accuracies'][i] += (in_bucket.float() * correct).sum()
#         bucket_data['confidences'][i] += (in_bucket.float() * max_probs).sum()
#
# # Calculate the average accuracy and confidence for each bucket
# for i in range(M):
#     if bucket_data['counts'][i] > 0:
#         bucket_data['accuracies'][i] /= bucket_data['counts'][i]
#         bucket_data['confidences'][i] /= bucket_data['counts'][i]
#
# # Calculate ECE
# ece = 0.0
# for i in range(M):
#     ece += (bucket_data['counts'][i] / total_cnt) * torch.abs(
#         bucket_data['accuracies'][i] - bucket_data['confidences'][i])
#
# print(total_cnt)
# print("Expected Calibration Error (ECE): {:.4f}".format(100*ece.item()))
#
# import matplotlib.pyplot as plt
#
# # Assuming bucket_data is already computed as in the provided code snippet
# # We will create a similar plot to the uploaded one using the calculated ECE and bucket data
#
# # Extract the average accuracy and average confidence for each bucket
# accuracies = bucket_data['accuracies'].cpu().numpy()
# confidences = bucket_data['confidences'].cpu().numpy()
# counts = bucket_data['counts'].cpu().numpy()
# total_count = counts.sum()
#
# # Normalize counts to get the proportion of data points in each bucket
# proportions = counts / total_count
#
# # Create the reliability diagram
# fig, ax = plt.subplots(figsize=(6, 6))
#
# # Add the bars indicating the confidence for each bin
# ax.bar(confidences, accuracies, width=1.0/M, edgecolor='black', alpha=0.5, label='Outputs')
#
# # Add the identity line
# ax.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
#
# # Annotate ECE on the plot
# ece_percentage = 100 * ece.item()  # ECE as a percentage
# # ax.text(0.1, 0.9, f'ECE: {ece_percentage:.2f}%', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
#
# # Set the limits and labels
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xlabel('Confidence')
# ax.set_ylabel('Accuracy')
#
# # Set the title and legend
# ax.set_title(f'Reliability Diagram: ECE: {ece_percentage:.2f}%')
# ax.legend()
#
# plt.show()
# plt.savefig(f'{RunningParams.prj_dir}/Reliability_Diagram_{set}_top1_HP_MODEL1_HP_FE.png')


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
save_file = f'{RunningParams.prj_dir}/faiss/advising_process_{set}_top1_HP_MODEL1_HP_FE.npy'
print(save_file)
print(set)
print(depth_of_pred)
np.save(save_file, faiss_nn_dict)
