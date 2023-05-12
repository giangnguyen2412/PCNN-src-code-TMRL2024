import random
from shutil import copyfile

from datasets import ImageFolderForNNs
from helpers import HelperFunctions
from params import RunningParams
from datasets import Dataset
import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets
import os

RunningParams = RunningParams()
Dataset = Dataset()
faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/Final_RN50_dataset_CUB_LP/train',
                                         transform=Dataset.data_transforms['train'])

faiss_data_loader = torch.utils.data.DataLoader(
    faiss_dataset,
    batch_size=RunningParams.batch_size,
    shuffle=False,  # turn shuffle to True
    num_workers=16,  # Set to 0 as suggested by
    # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
    pin_memory=True,
)
INDEX_FILE = '../faiss/Final_faiss_CUB200_class_idx_dict_HP_extractor.npy'

faiss_nns_class_dict = np.load(INDEX_FILE, allow_pickle="False", ).item()
targets = faiss_data_loader.dataset.targets
faiss_data_loader_ids_dict = dict()
faiss_loader_dict = dict()
for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
    faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if
                                            targets[x] == class_id]  # check this value
    class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
    class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
    faiss_loader_dict[class_id] = class_id_loader


HelperFunctions = HelperFunctions()
filename = '../faiss/clean_Final_faiss_CUB_final_train_top1_HP_MODEL1_HP_FE.npy'
npy_file = np.load(filename, allow_pickle=True, ).item()

write_filename = '../faiss/clean_Final_faiss_CUB_final_train_top1_HP_MODEL1_HP_FE.npy'
write_npy_file = np.load(write_filename, allow_pickle=True, ).item()

crt, wrong = 0, 0
cnt = 0
for file, nns in npy_file.items():
    nn0 = nns[0]
    nn0_basename = nn0.split('/')[-2]
    nn_class_name = nn0_basename.split('.')[1]
    if 'aug' in file:
        continue
    # if nn_class_name.lower() in file.lower():
    #     nn = '/home/giang/Downloads/Final_RN50_dataset_CUB_HP/final_train/{}/{}'.format(nn0_basename, file)
    #     new_file = 'crt_' + file
    #     nn = [nn]*6
    #     crt += 1
    # else:
    if nn_class_name.lower() not in file.lower():
        new_file = 'wrong2_' + file
        class_id = faiss_data_loader.dataset.class_to_idx[nn0_basename]
        nns_candidate = [x for x in range(len(targets)) if
                                                targets[x] == class_id]  # check this value
        random.shuffle(nns_candidate)
        wrong += 1
        nn = []
        for i in range(6):
            nn_id = nns_candidate[i]
            nn.append(faiss_data_loader.dataset.imgs[nn_id][0])

        gt_wnid = [x for x in faiss_data_loader.dataset.classes if x.split('.')[1].lower() in file.lower()]
        src_file = os.path.join('/home/giang/Downloads/Final_RN50_dataset_CUB_HP/clean_train', gt_wnid[0], file)
        dst_file = os.path.join('/home/giang/Downloads/Final_RN50_dataset_CUB_HP/clean_train', gt_wnid[0], new_file)

        copyfile(src_file, dst_file)
        if len(gt_wnid) > 0:
            cnt += 1

        write_npy_file[new_file] = nn

print(crt, wrong)
print(cnt)
print(len(write_npy_file))
np.save('../faiss/new_Final_faiss_CUB_final_train_top1_HP_MODEL1_HP_FE.npy', write_npy_file)

