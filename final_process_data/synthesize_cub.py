import random
import os
from shutil import copyfile
# Set the random seed to ensure reproducibility
random.seed(42)

# Set the paths to the source and destination directories
source_dir = "/home/giang/Downloads/Final_RN50_dataset_CUB_LP/val"
dest_dir = "/home/giang/Downloads/NeurIPS22_CUB/final_train"

# Get a list of the subfolders in the destination directory
interest_folders = os.listdir(dest_dir)

abs_paths = []
all_files = []

for subfolder in interest_folders:
    # Construct the full path to the source subdirectory
    source_subdir_path = os.path.join(source_dir, subfolder)

    # Get a list of the .jpg files in the source subdirectory
    files = [file for file in os.listdir(source_subdir_path) if file.endswith(".jpg")]
    for file in files:
        abs_paths.append(os.path.join(source_subdir_path, file))
        all_files.append(file)

cnt = 0
for idx, file in enumerate(all_files):
    abs_path = abs_paths[idx]
    src_file = abs_path
    class_name = os.path.basename(os.path.dirname(src_file))
    dst_file = os.path.join(dest_dir, class_name, file)
    cnt += 1
    copyfile(src_file, dst_file)

print(cnt)