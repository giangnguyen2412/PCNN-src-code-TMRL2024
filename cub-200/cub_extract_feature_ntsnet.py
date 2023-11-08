import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import os
import faiss
from PIL import Image

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from torchvision import datasets, models, transforms
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


Dataset = Dataset()
RunningParams = RunningParams()

import os
import torch.utils.data
from torch.nn import DataParallel
from core import model, dataset
from torch import nn
from tqdm import tqdm
net = model.attention_net(topN=6)
ckpt = torch.load(f'{RunningParams.parent_dir}/NTS-Net/model.ckpt')

net.load_state_dict(ckpt['net_state_dict'])

net.eval()
net = net.cuda()
net = DataParallel(net)
MODEL1 = net
MODEL1.eval()

attention_net = list(MODEL1.children())
components = list(attention_net[0].children())
resnet = components[0]
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)

in_features = 2048
print("Building FAISS index...! Training set is the knowledge base.")

data_transforms = transforms.Compose([
    transforms.Resize((600, 600), interpolation=Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

faiss_dataset = datasets.ImageFolder(f'{RunningParams.parent_dir}/datasets/CUB/advnet/train',
                                     transform=data_transforms)

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)

INDEX_FILE = 'faiss/cub/NTS-NET.npy'
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


net = model.attention_net(topN=6)
ckpt = torch.load(f'{RunningParams.parent_dir}/NTS-Net/model.ckpt')

net.load_state_dict(ckpt['net_state_dict'])

net.eval()
net = net.cuda()
net = DataParallel(net)
MODEL1 = net
MODEL1 = nn.DataParallel(MODEL1).eval()

set = 'test'
data_dir = '{}/datasets/CUB/advnet/{}'.format(RunningParams.parent_dir, set)

image_datasets = dict()
image_datasets['train'] = ImageFolderWithPaths(data_dir, data_transforms)
train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=16,
    shuffle=False,  # Don't turn shuffle to False --> model works wrongly
    num_workers=16,
    pin_memory=True,
)

depth_of_pred = 1
correct_cnt = 0
total_cnt = 0

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

    _, out, _, _, _ = MODEL1(data.cuda())
    model1_p = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.topk(model1_p, depth_of_pred, dim=1)
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
                _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value)

                for id in range(indices.shape[1]):
                    id = loader.dataset.indices[indices[0, id]]
                    nn_list.append(loader.dataset.dataset.imgs[id][0])
                faiss_nn_dict[base_name] = nn_list
            else:

                if i == 0:  # top-1 predictions --> Enrich top-1 prediction samples
                    _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), faiss_index.ntotal)

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
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value+1)
                        indices = indices[:, 1:]  # skip the first NN
                    else:
                        key = 'Wrong_{}_'.format(i) + base_name
                        _, indices = faiss_index.search(embeddings[sample_idx].reshape([1, in_features]), RunningParams.k_value)

                    for id in range(indices.shape[1]):
                        id = loader.dataset.indices[indices[0, id]]
                        nn_list.append(loader.dataset.dataset.imgs[id][0])

                    faiss_nn_dict[key] = dict()
                    faiss_nn_dict[key]['NNs'] = nn_list
                    faiss_nn_dict[key]['label'] = int(predicted_idx == gt_id)
                    faiss_nn_dict[key]['conf'] = score[sample_idx][i].item()

# print("Top-1 Accuracy: {}".format(correct_cnt * 100 / total_cnt))
np.save('faiss/cub/NTSNet_{}_{}_{}.npy'.format(depth_of_pred, RunningParams.k_value, set),
        faiss_nn_dict)
