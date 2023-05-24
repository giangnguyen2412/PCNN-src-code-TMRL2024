# import torch
# import torch.nn as nn
#
# import numpy as np
# from datasets import ImageFolderForNNs
# from helpers import HelperFunctions
# import os
# import glob
#
# filename = 'faiss/cub/top2_NeurIPS_Finetuning_faiss_CUB_train_all_top1_HP_MODEL1_HP_FE.npy'
# kbc = np.load(filename, allow_pickle=True, ).item()
import os
import numpy as np
import shutil
from shutil import copyfile

# Set the paths and filenames
folder_path = "/home/giang/Downloads/datasets/CUB_train_all_top3_final/"
dict_path = "faiss/cub/top3_final_NeurIPS_Finetuning_faiss_CUB_train_all_top1_HP_MODEL1_HP_FE.npy"

# Load the dictionary
file_dict = np.load(dict_path, allow_pickle=True).item()

import os

cnt = 0
# Function to recursively find and print JPG files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".jpg"):
            file_path = os.path.join(root, file)
            crt_file = 'Correct_' + file
            if crt_file in file_dict:
                src_path = file_path
                dst_path = os.path.join(root, crt_file)
                copyfile(src_path, dst_path)
                cnt +=1

            crt_file = 'Correct_Correct_' + file
            if crt_file in file_dict:
                src_path = file_path
                dst_path = os.path.join(root, crt_file)
                copyfile(src_path, dst_path)
                cnt += 1

            wrong_file = 'Wrong_' + file
            if wrong_file in file_dict:
                src_path = file_path
                dst_path = os.path.join(root, wrong_file)
                copyfile(src_path, dst_path)
                cnt +=1

            wrong_file = 'Wrong_Wrong_' + file
            if wrong_file in file_dict:
                src_path = file_path
                dst_path = os.path.join(root, wrong_file)
                copyfile(src_path, dst_path)
                cnt += 1

            os.remove(file_path)

print(cnt)
print("Files renamed/copied successfully!")
