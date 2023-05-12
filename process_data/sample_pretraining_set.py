import os
import shutil
import numpy as np

# Load the dictionary
finetuning = np.load('../Finetuning.npy', allow_pickle=True).item()

# Get the list of files in the test folder
test_files = finetuning['final_test'] + finetuning['final_val'] + finetuning['final_train']

# Define the paths for the train_tmp and val_tmp folders
train_tmp_path = '/home/giang/Downloads/Final_RN50_dataset_CUB_LP/train'
val_tmp_path = '/home/giang/Downloads/Final_RN50_dataset_CUB_LP/val'

# Create the train_tmp and val_tmp folders if they don't exist
os.makedirs(train_tmp_path, exist_ok=True)
os.makedirs(val_tmp_path, exist_ok=True)
all_files = []

# Loop over the subfolders in the combined folder
for folder_name in os.listdir('/home/giang/Downloads/datasets/CUB/combined'):

    # Skip non-directory files
    if not os.path.isdir(os.path.join('/home/giang/Downloads/datasets/CUB/combined', folder_name)):
        continue

    # # Create the corresponding train_tmp and val_tmp subfolders
    # os.makedirs(os.path.join(train_tmp_path, folder_name), exist_ok=True)
    # os.makedirs(os.path.join(val_tmp_path, folder_name), exist_ok=True)

    # Get the list of files in the current folder, excluding the test files
    files = os.listdir(os.path.join('/home/giang/Downloads/datasets/CUB/combined', folder_name))
    files = [[folder_name, f] for f in files if f not in test_files]
    all_files.extend(files)


np.random.seed(42)
# Shuffle the files and split them into train and val
np.random.shuffle(all_files)
train_files = all_files[:8000]
val_files = all_files[8000:]

# Move the train files to the train_tmp folder
for folder_name, file_name in train_files:
    src_path = os.path.join('/home/giang/Downloads/datasets/CUB/combined', folder_name, file_name)
    os.makedirs(os.path.join(train_tmp_path, folder_name), exist_ok=True)
    dst_path = os.path.join(train_tmp_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)

# Move the val files to the val_tmp folder
for folder_name, file_name in val_files:
    src_path = os.path.join('/home/giang/Downloads/datasets/CUB/combined', folder_name, file_name)
    os.makedirs(os.path.join(val_tmp_path, folder_name), exist_ok=True)
    dst_path = os.path.join(val_tmp_path, folder_name, file_name)
    shutil.copy(src_path, dst_path)