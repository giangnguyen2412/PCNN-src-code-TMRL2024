import os
import shutil
import random

# Set the path to the original folder containing all the subfolders
original_folder = '/home/giang/Downloads/RN50_dataset_CUB_HP/merged'

# Set the path to the new folders where we will move the files
train_folder = '/home/giang/Downloads/RN50_dataset_CUB_HP/train'
val_folder = '/home/giang/Downloads/RN50_dataset_CUB_HP/val'
test_folder = '/home/giang/Downloads/RN50_dataset_CUB_HP/test'

# Set the percentage split for training, validation, and test data
train_split = 1000
val_split = 320
test_split = 320

# Loop through each subfolder in the original folder
for subfolder in os.listdir(original_folder):
    subfolder_path = os.path.join(original_folder, subfolder)

    # Create the corresponding subfolders in the train, val, and test folders
    train_subfolder_path = os.path.join(train_folder, subfolder)

    val_subfolder_path = os.path.join(val_folder, subfolder)

    test_subfolder_path = os.path.join(test_folder, subfolder)

    # Loop through each file in the subfolder
    files = os.listdir(subfolder_path)
    random.shuffle(files)
    n = len(files)

    train_end = int(n * train_split/(train_split+val_split+test_split))
    val_end = train_end + int(n * val_split/(train_split+val_split+test_split))
    test_end = val_end + int(n * test_split/(train_split+val_split+test_split))

    for i, file_name in enumerate(files):
        file_path = os.path.join(subfolder_path, file_name)

        # Copy the file to the appropriate folder based on the split percentages
        if i < train_end:
            os.makedirs(train_subfolder_path, exist_ok=True)
            shutil.copy(file_path, train_subfolder_path)
        elif i < val_end:
            os.makedirs(val_subfolder_path, exist_ok=True)
            shutil.copy(file_path, val_subfolder_path)
        else:
            os.makedirs(test_subfolder_path, exist_ok=True)
            shutil.copy(file_path, test_subfolder_path)
