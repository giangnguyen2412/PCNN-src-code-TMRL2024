# Extract NNs for training AdvNets
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import faiss

from tqdm import tqdm
from torchvision import datasets, models, transforms
import sys
sys.path.insert(0, '/home/giang/Downloads/advising_network')

from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

Dataset = Dataset()
RunningParams = RunningParams('CUB')

if RunningParams.VisionTransformer is True:
    import timm
    from torchvision import datasets, transforms


    class CustomViT(nn.Module):
        def __init__(self, base_model):
            super(CustomViT, self).__init__()
            self.base_model = base_model

        def forward(self, x):
            # Get the features from the base ViT model
            x = self.base_model.forward_features(x)
            # Extract the CLS token (first token)
            cls_token = x[:, 0]
            # Pass the features through the classifier
            output = self.base_model.head(cls_token)
            return output, cls_token

    # Initialize the base model and load the trained weights
    base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=200)
    model_path = "./vit_base_patch16_224_cub_200way.pth"
    state_dict = torch.load(model_path, map_location=torch.device("cuda"))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict)

    # Wrap the base model in the custom model
    model = CustomViT(base_model)
    MODEL1 = model.cuda()
else:

    from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')

    # TODO: Define INAT and IMAGENET features to be used in two cases of using RN50

    if RunningParams.resnet == 50 and RunningParams.RN50_INAT is False:
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet50_cub_200way_top1acc_63.pth')
    elif RunningParams.resnet == 34:
        resnet = models.resnet34(pretrained=True)
        resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet34_cub_200way_top1acc_62_81.pth')
    elif RunningParams.resnet == 18:
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet18_cub_200way_top1acc_60_22.pth')

    resnet.load_state_dict(my_model_state_dict, strict=True)
    if RunningParams.resnet == 34 or RunningParams.resnet == 18 or (
            RunningParams.resnet == 50 and RunningParams.RN50_INAT is False):
        resnet.fc = resnet.fc[0]

    MODEL1 = resnet.cuda()

MODEL1.eval()

if RunningParams.VisionTransformer is True:
    feature_extractor = MODEL1
else:
    # Conv layers and the avg pooling layer
    feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature

feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

if RunningParams.VisionTransformer is True:
    in_features = 768
else:
    if RunningParams.resnet == 34 or RunningParams.resnet == 18:
        in_features = resnet.fc.in_features
    elif RunningParams.resnet == 50:
        in_features = 2048

print("Building FAISS index...! Training set is the knowledge base.")

# faiss dataset contains images using as the knowledge based for KNN retrieval
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

if RunningParams.VisionTransformer is True:
    INDEX_FILE = f'{RunningParams.prj_dir}/faiss/cub/ViT_faiss_CUB200_class_idx_dict_HP_extractor.npy'
else:
    INDEX_FILE = f'{RunningParams.prj_dir}/faiss/cub/INAT{RunningParams.RN50_INAT}_RN{RunningParams.resnet}_faiss_CUB200_class_idx_dict_HP_extractor.npy'

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

            if RunningParams.VisionTransformer is True:
                _, embeddings = feature_extractor(data.cuda()) # 768 for ViT-B-16
            else:
                embeddings = feature_extractor(data.cuda())
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


# from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck
#
# resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
# my_model_state_dict = torch.load(
#     f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')
# resnet.load_state_dict(my_model_state_dict, strict=True)
# MODEL1 = resnet.cuda()
# MODEL1.eval()
MODEL1 = nn.DataParallel(MODEL1).eval()

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(RunningParams.data_dir, Dataset.data_transforms['train'])
print(RunningParams.data_dir)
train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=128,
    shuffle=False,  # Don't turn shuffle to False --> model works wrongly
    num_workers=16,
    pin_memory=True,
)

depth_of_pred = RunningParams.Q

# if RunningParams.set == 'test':
#     depth_of_pred = 1
correct_cnt = 0
total_cnt = 0

faiss_nn_dict = dict()
for batch_idx, (data, label, paths) in enumerate(tqdm(train_loader)):
    if len(train_loader.dataset.classes) < 200:
        for sample_idx in range(data.shape[0]):
            tgt = label[sample_idx].item()
            class_name = train_loader.dataset.classes[tgt]
            id = faiss_dataset.class_to_idx[class_name]
            label[sample_idx] = id

    if RunningParams.VisionTransformer is True:
        out, embeddings = feature_extractor(data.cuda())  # 768 for ViT-B-16
    else:
        embeddings = feature_extractor(data.cuda())
        embeddings = torch.flatten(embeddings, start_dim=1)

    embeddings = embeddings.cpu().detach().numpy()

    if RunningParams.VisionTransformer is True:
        pass
    else:
        out = MODEL1(data.cuda())

    model1_p = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.topk(model1_p, depth_of_pred, dim=1)

    # Top-1 prediction is the first element in each row of 'index'
    top1_pred = index[:, 0]

    # Increment correct count if prediction matches label
    correct_cnt += (top1_pred == label.cuda()).sum().item()
    total_cnt += label.size(0)

    for sample_idx in range(data.shape[0]):
        base_name = os.path.basename(paths[sample_idx])
        gt_id = label[sample_idx]

        for i in range(depth_of_pred):
            # Get the top-k predicted label
            predicted_idx = index[sample_idx][i].item()

            # Dataloader and knowledge base upon the predicted class
            loader = faiss_loader_dict[predicted_idx]
            faiss_index = faiss_nns_class_dict[predicted_idx]
            nn_list = list()

            if depth_of_pred == 1:  # For val and test sets
                _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.negative_order)

                indices = indices[:, RunningParams.negative_order - 1:]

                for id in range(indices.shape[1]):
                    id = loader.dataset.indices[indices[0, id]]
                    nn_list.append(loader.dataset.dataset.imgs[id][0])

                key = base_name
                # faiss_nn_dict[base_name] = nn_list

                faiss_nn_dict[key] = dict()
                faiss_nn_dict[key]['NNs'] = nn_list
                faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()
                faiss_nn_dict[key]['input_gt'] = loader.dataset.dataset.classes[gt_id.item()]
            else:

                if i == 0:  # top-1 predictions --> Enrich top-1 prediction samples --> value Q
                    _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), faiss_index.ntotal)

                    for j in range(depth_of_pred):  # Make up x NN sets from top-1 predictions
                        nn_list = list()

                        if predicted_idx == gt_id:
                            key = 'Correct_{}_{}_'.format(i, j) + base_name
                            min_id = (j * RunningParams.k_value) + 1  # 3 NNs for one NN set
                            max_id = ((j * RunningParams.k_value) + RunningParams.k_value) + 1
                        else:
                            key = 'Wrong_{}_{}_'.format(i, j) + base_name
                            min_id = j * RunningParams.k_value  + RunningParams.negative_order-1 # 3 NNs for one NN set ### ----
                            max_id = (j * RunningParams.k_value) + RunningParams.k_value + RunningParams.negative_order-1  ### ------------

                        for id in range(min_id, max_id):
                            id = loader.dataset.indices[indices[0, id]]
                            nn_list.append(loader.dataset.dataset.imgs[id][0])

                        faiss_nn_dict[key] = dict()
                        faiss_nn_dict[key]['NNs'] = nn_list
                        faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                        faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()
                        faiss_nn_dict[key]['input_gt'] = loader.dataset.dataset.classes[gt_id.item()]

                else:
                    if predicted_idx == gt_id:
                        key = 'Correct_{}_'.format(i) + base_name
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.negative_order)
                        indices = indices[:, 1:]  # skip the first NN
                    else:
                        key = 'Wrong_{}_'.format(i) + base_name
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.negative_order)

                        indices = indices[:, RunningParams.negative_order-1:]  # skip the first NN -----------

                    for id in range(indices.shape[1]):
                        id = loader.dataset.indices[indices[0, id]]
                        nn_list.append(loader.dataset.dataset.imgs[id][0])

                    faiss_nn_dict[key] = dict()
                    faiss_nn_dict[key]['NNs'] = nn_list
                    faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                    faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()
                    faiss_nn_dict[key]['input_gt'] = loader.dataset.dataset.classes[gt_id.item()]

# Calculate top-1 accuracy
top1_accuracy = correct_cnt / total_cnt * 100
print(f"Top-1 {RunningParams.set} accuracy of the Classifier: {top1_accuracy:.2f}%")

file_name = RunningParams.faiss_npy_file
np.save(file_name, faiss_nn_dict)
print(len(faiss_nn_dict))
print(file_name)

empty_cnt = 0
for k, v in faiss_nn_dict.items():
    # breakpoint()
    if len(v['NNs']) == 0:
        empty_cnt += 1
        continue
    NN_class = os.path.basename(os.path.dirname(v['NNs'][0]))
    if v['label'] == 1:
        if NN_class.lower() in v['input_gt'].lower():
            continue
        else:
            print('You retrieved wrong NNs. The label for the pair is positive but two images are from different classes!')
            exit(-1)
    else:
        if NN_class.lower() not in v['input_gt'].lower():
            continue
        else:
            print('You retrieved wrong NNs. The label for the pair is negative but two images are from same class!')
            exit(-1)

print('Passed sanity checks for extracting NNs!')
print(f'faiss NULL retrieval count: {empty_cnt}')

################################################################

new_dict = dict()
for k, v in faiss_nn_dict.items():
    for nn in v['NNs']:
        base_name = os.path.basename(nn)
        if base_name in k:
            break
        else:
            new_dict[k] = v
np.save(file_name, new_dict)
print(file_name)
print('DONE: Cleaning duplicates entries: query and NN being similar!')

################################################################

import shutil
import os

source_folder = RunningParams.data_dir
destination_folder = RunningParams.aug_data_dir

if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)

# Check if the destination folder already exists
if os.path.exists(destination_folder):
    print(f"Destination folder '{destination_folder}' already exists. Exiting to avoid overwriting.")
else:
    shutil.copytree(source_folder, destination_folder)
    print(f"Copied '{source_folder}' to '{destination_folder}'")

