import numpy as np
from datasets import ImageFolderForNNs
from helpers import HelperFunctions
import os
import glob

filename = 'faiss/faiss_CUB_val_top1_LP_MODEL1_HP_FE.npy'
kbc = np.load(filename, allow_pickle=True, ).item()

pass
# import glob
# images = glob.glob('/home/giang/Downloads/datasets/CUB/combined_2/*/*.*')
#
# cnt = 0
# files = []
# for key, item in kbc.items():
#     file_name = key
#     for img in images:
#         if file_name in img:
#             cnt += 1
#             files.append(img)
#             break
#
# print(cnt)
# # print(img)
#
# import os
# import shutil
#
# # Define the list of file paths
# file_paths = files  # Replace [...] with your list of file paths
#
# # Define the destination directory
# destination_dir = '/home/giang/Downloads/RN50_dataset_CUB_Pretraining/val'
#
# # Loop through each file path in the list
# for file_path in file_paths:
#     # Extract the directory path from the file path
#     dir_path = os.path.dirname(file_path)
#     dir_path = os.path.basename(dir_path)
#
#     # Construct the destination directory path
#     dest_dir_path = os.path.join(destination_dir, dir_path)
#
#     # Create the destination directory if it doesn't exist
#     os.makedirs(dest_dir_path, exist_ok=True)
#
#     # Construct the destination file path
#     dest_file_path = os.path.join(dest_dir_path, os.path.basename(file_path))
#
#     # Move the file to the destination directory
#     shutil.move(file_path, dest_file_path)


# import os
# import random
# import shutil
#
# source_folder = "/home/giang/Downloads/RN50_dataset_CUB_HIGH/val/Correct"
# destination_folder = "/home/giang/Downloads/RN50_dataset_CUB_Finetuning/val"
# num_files = 145
#
# # Create destination folder if it doesn't exist
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#
# # Loop through subfolders in the source folder
# for dirpath, dirnames, filenames in os.walk(source_folder):
#     # Determine the corresponding subfolder in the destination folder
#     relative_path = os.path.relpath(dirpath, source_folder)
#     dest_dir = os.path.join(destination_folder, relative_path)
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
#
#     # Randomly choose up to num_files files to copy from this subfolder
#     files_to_copy = random.sample(filenames, min(num_files, len(filenames)))
#
#     # Copy the selected files to the corresponding subfolder in the destination folder
#     for filename in files_to_copy:
#         source_file = os.path.join(dirpath, filename)
#         dest_file = os.path.join(dest_dir, filename)
#         shutil.copy2(source_file, dest_file)
#
#         num_files -= 1
#         if num_files == 0:
#             break
#
#     if num_files == 0:
#         break
#
# print("Random files copied to: ", destination_folder)

