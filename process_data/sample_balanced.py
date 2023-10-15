import os
import shutil
import random

# Set the path to the original folder containing all the subfolders
original_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/merged'

# Set the path to the new folders where we will move the files
train_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/tmp_train'
val_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/tmp_val'
test_folder = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/tmp_test'

ignore = 0
train_cnt, val_cnt, test_cnt = 0, 0, 0
train_cycle_limit, val_cycle_limit, test_cycle_limit = 3, 1 , 1

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

    if n < 5:
        ignore += 1
        continue

    train_ratio_cnt = 0

    train, val, test = True, False, False
    for i, file_name in enumerate(files):
        file_path = os.path.join(subfolder_path, file_name)

        # Copy the file to the appropriate folder based on the split percentages
        if train is True:
            os.makedirs(train_subfolder_path, exist_ok=True)
            shutil.copy(file_path, train_subfolder_path)

            train_ratio_cnt += 1

            if train_ratio_cnt == train_cycle_limit:
                train, val, test = False, True, False
                train_ratio_cnt = 0

            train_cnt += 1
            continue

        elif val is True:
            os.makedirs(val_subfolder_path, exist_ok=True)
            shutil.copy(file_path, val_subfolder_path)

            train, val, test = False, False, True
            val_cnt += 1
            continue

        elif test is True:
            os.makedirs(test_subfolder_path, exist_ok=True)
            shutil.copy(file_path, test_subfolder_path)

            train, val, test = True, False, False
            test_cnt += 1
            continue

print("Ignored {} categories".format(ignore))
print("Train:{} Val:{} Test:{}".format(train_cnt, val_cnt, test_cnt))