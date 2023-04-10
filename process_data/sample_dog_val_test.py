import os
import shutil
import random

# Set the path to the original folder containing all the subfolders
original_folder = '/home/giang/Downloads/datasets/Dogs_val'

# Set the path to the new folders where we will move the files
val_folder = '/home/giang/Downloads/datasets/Dogs_val_tmp'
test_folder = '/home/giang/Downloads/datasets/Dogs_test_tmp'

# Set the percentage split for training, validation, and test data
val_split = 1084
test_split = 1084

# Loop through each subfolder in the original folder
for subfolder in os.listdir(original_folder):
    subfolder_path = os.path.join(original_folder, subfolder)

    val_subfolder_path = os.path.join(val_folder, subfolder)

    test_subfolder_path = os.path.join(test_folder, subfolder)

    # Loop through each file in the subfolder
    files = os.listdir(subfolder_path)
    random.shuffle(files)
    n = len(files)

    val_end = int(n * val_split/(val_split+test_split))
    test_end = val_end + int(n * test_split/(val_split+test_split))

    for i, file_name in enumerate(files):
        file_path = os.path.join(subfolder_path, file_name)

        # Copy the file to the appropriate folder based on the split percentages
        if i < val_end:
            os.makedirs(val_subfolder_path, exist_ok=True)
            shutil.copy(file_path, val_subfolder_path)
        else:
            os.makedirs(test_subfolder_path, exist_ok=True)
            shutil.copy(file_path, test_subfolder_path)
