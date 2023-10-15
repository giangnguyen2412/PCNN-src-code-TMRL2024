import os
import shutil
import random

import os
import shutil

################################################################
# Merge two folders train/ and val/
################################################################

def merge_folders(src_folder, dest_folder):
    # Create the destination directory if it doesn't already exist

    # Recursively walk through the source directory
    for root, dirs, files in os.walk(src_folder):
        # For each file, get the full path of the source file and the destination file
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_folder, os.path.relpath(src_file_path, src_folder))
            # Copy the file to the destination directory while maintaining the parent folder structure
            parent_folder = os.path.relpath(src_file_path, src_folder).split('/')[0]
            parent_folder = os.path.join(dest_folder, parent_folder)
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            shutil.copy2(src_file_path, dest_file_path)

train_dir = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/val_tmp/'
val_dir = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/test_tmp/'

merged_dir = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/tmp/'
merge_folders(train_dir, merged_dir)
merge_folders(val_dir, merged_dir)

################################################################
# Randomly split train and val to 83/13
################################################################
# source_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/merged'
# train_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/train_tmp'
# val_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/val_tmp'
#
# train_ratio = 0.9  # adjust as needed
# file_list = []
#
# # recursively find all jpg files in source folder
# for root, dirs, files in os.walk(source_folder):
#     for file in files:
#         if file.endswith('.jpg'):
#             file_list.append(os.path.join(root, file))
#
# # randomly shuffle the file list
# random.shuffle(file_list)
#
# # split the file list into train and val lists based on train_ratio
# train_files = file_list[:1200]
# val_files = file_list[1200:]
#
# # create train and val folders if they don't exist
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)
#
# # copy train files to train folder, maintaining subfolder structure
# for file_path in train_files:
#     subfolder_path = os.path.join(train_folder, os.path.relpath(os.path.dirname(file_path), source_folder))
#     os.makedirs(subfolder_path, exist_ok=True)
#     shutil.copy(file_path, subfolder_path)
#
# # copy val files to val folder, maintaining subfolder structure
# for file_path in val_files:
#     subfolder_path = os.path.join(val_folder, os.path.relpath(os.path.dirname(file_path), source_folder))
#     os.makedirs(subfolder_path, exist_ok=True)
#     shutil.copy(file_path, subfolder_path)
