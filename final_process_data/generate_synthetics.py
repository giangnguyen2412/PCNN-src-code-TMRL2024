import random
from shutil import copyfile
import numpy as np
from tqdm import tqdm
from torchvision import datasets
import os

import glob

correctness_bins = ['Correct', 'Wrong']
category_dict = {}
categorized_path = '/home/giang/Downloads/RN50_dataset_CUB_HIGH/combined'

for c in correctness_bins:
    dir = os.path.join(categorized_path, c)
    files = glob.glob(os.path.join(dir, '*', '*.*'))
    key = c
    for file in files:
        base_name = os.path.basename(file)
        category_dict[base_name] = key

dest_dir = "/home/giang/Downloads/NeurIPS22_CUB/final_train"

# Get a list of the subfolders in the destination directory
interest_folders = os.listdir(dest_dir)

abs_paths = []
all_files = []

for subfolder in interest_folders:
    # Construct the full path to the source subdirectory
    source_subdir_path = os.path.join(dest_dir, subfolder)

    # Get a list of the .jpg files in the source subdirectory
    files = [file for file in os.listdir(source_subdir_path) if file.endswith(".jpg")]
    for file in files:
        abs_paths.append(os.path.join(source_subdir_path, file))
        all_files.append(file)

crt, wrong = 0, 0
cnt = 0
for idx, file in enumerate(all_files):
    abs_path = abs_paths[idx]
    class_name = os.path.basename(os.path.dirname(abs_path))

    if category_dict[file] == 'Correct':
        new_file = 'crt_' + file
        crt += 1

        src_file = os.path.join('/home/giang/Downloads/NeurIPS22_CUB/final_train', class_name, file)
        dst_file = os.path.join('/home/giang/Downloads/NeurIPS22_CUB/final_train', class_name, new_file)
        copyfile(src_file, dst_file)

    elif category_dict[file] == 'Wrong':
        for i in range(21):
            new_file = 'wrong{}_'.format(i) + file
            wrong += 1

            src_file = os.path.join('/home/giang/Downloads/NeurIPS22_CUB/final_train', class_name, file)
            dst_file = os.path.join('/home/giang/Downloads/NeurIPS22_CUB/final_train', class_name, new_file)
            copyfile(src_file, dst_file)

    else:
        print('Something wrong!')
        exit(-1)


print(crt, wrong)

