import os
import shutil
import numpy as np


# Define the paths for the train_tmp and val_tmp folders
train_tmp_path = '/home/giang/Downloads/datasets/CUB_pre_train'
val_tmp_path = '/home/giang/Downloads/datasets/CUB_pre_val'

# Create the train_tmp and val_tmp folders if they don't exist
os.makedirs(train_tmp_path, exist_ok=True)
os.makedirs(val_tmp_path, exist_ok=True)
all_files = []

src_dir = '/home/giang/Downloads/datasets/CUB_pre'

# Loop over the subfolders in the combined folder
for folder_name in os.listdir(src_dir):

    # Skip non-directory files
    if not os.path.isdir(os.path.join(src_dir, folder_name)):
        continue

    # Get the list of files in the current folder, excluding the test files
    files = os.listdir(os.path.join(src_dir, folder_name))
    files = [[folder_name, f] for f in files]
    all_files.extend(files)

np.random.seed(42)
# Shuffle the files and split them into train and val
np.random.shuffle(all_files)
train_files = all_files[:9000]
val_files = all_files[9000:]

# Move the train files to the train_tmp folder
for folder_name, file_name in train_files:
    src_path = os.path.join(src_dir, folder_name, file_name)
    os.makedirs(os.path.join(train_tmp_path, folder_name), exist_ok=True)
    dst_path = os.path.join(train_tmp_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)

# Move the val files to the val_tmp folder
for folder_name, file_name in val_files:
    src_path = os.path.join(src_dir, folder_name, file_name)
    os.makedirs(os.path.join(val_tmp_path, folder_name), exist_ok=True)
    dst_path = os.path.join(val_tmp_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)