import os
import shutil
import numpy as np

# Define the paths for the train_tmp and val_tmp folders
train_path = '/home/giang/Downloads/datasets/CUB_train'
val_path = '/home/giang/Downloads/datasets/CUB_val'
test_path = '/home/giang/Downloads/datasets/CUB_test'


# Create the train_tmp and val_tmp folders if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
all_files = []

# Loop over the subfolders in the combined folder
for folder_name in os.listdir('/home/giang/Downloads/datasets/CUB/test0'):

    # Skip non-directory files
    if not os.path.isdir(os.path.join('/home/giang/Downloads/datasets/CUB/test0', folder_name)):
        continue

    # Get the list of files in the current folder, excluding the test files
    files = os.listdir(os.path.join('/home/giang/Downloads/datasets/CUB/test0', folder_name))
    tmp_list = []
    for file in files:
        tmp_list.append([folder_name, file])
    all_files.extend(tmp_list)


# Shuffle the files and split them into train and val
np.random.shuffle(all_files)
train_files = all_files[:3794]
val_files = all_files[3794:4794]
test_files = all_files[4794:]

# Move the train files to the train_tmp folder
for folder_name, file_name in train_files:
    src_path = os.path.join('/home/giang/Downloads/datasets/CUB/test0', folder_name, file_name)
    os.makedirs(os.path.join(train_path, folder_name), exist_ok=True)
    dst_path = os.path.join(train_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)

# Move the val files to the val_tmp folder
for folder_name, file_name in val_files:
    src_path = os.path.join('/home/giang/Downloads/datasets/CUB/test0', folder_name, file_name)
    os.makedirs(os.path.join(val_path, folder_name), exist_ok=True)
    dst_path = os.path.join(val_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)

# Move the val files to the val_tmp folder
for folder_name, file_name in test_files:
    src_path = os.path.join('/home/giang/Downloads/datasets/CUB/test0', folder_name, file_name)
    os.makedirs(os.path.join(test_path, folder_name), exist_ok=True)
    dst_path = os.path.join(test_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)
