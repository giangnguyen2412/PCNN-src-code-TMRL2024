# import os
# import shutil
#
# # Define the paths to the source and destination directories
# source_dir = "/home/giang/Downloads/RN50_dataset_CUB_HP/final/tmp_train"
# dest_dir = "/home/giang/Downloads/RN50_dataset_CUB_HP/final/tmp_test"
#
# # Loop over the subdirectories in the source directory
# for subdir in os.listdir(source_dir):
#     # Construct the full paths to the source and destination subdirectories
#     source_subdir_path = os.path.join(source_dir, subdir)
#     dest_subdir_path = os.path.join(dest_dir, subdir)
#
#     # Create the destination subdirectory if it doesn't already exist
#     if not os.path.exists(dest_subdir_path):
#         os.makedirs(dest_subdir_path)
#
#     # Get a list of the .jpg files in the source subdirectory
#     jpg_files = [file for file in os.listdir(source_subdir_path) if file.endswith(".jpg")]
#
#     # If there is at least one .jpg file in the source subdirectory, copy the first one to the destination subdirectory
#     if len(jpg_files) > 0:
#         first_jpg_file = jpg_files[0]
#         source_file_path = os.path.join(source_subdir_path, first_jpg_file)
#         dest_file_path = os.path.join(dest_subdir_path, first_jpg_file)
#         # shutil.move(source_file_path, dest_file_path)

############################################################################


import os

# Set the path to the directory to search
dir_path = "/home/giang/Downloads/Final_RN50_dataset_CUB_HP/"

# Initialize an empty list to hold the file names
jpg_files = []

# Loop over the subfolders in the top-level directory
for subfolder in os.listdir(dir_path):
    # Construct the full path to the subfolder
    subfolder_path = os.path.join(dir_path, subfolder)

    # Loop over the subsubfolders in the subfolder
    for subsubfolder in os.listdir(subfolder_path):
        # Construct the full path to the subsubfolder
        subsubfolder_path = os.path.join(subfolder_path, subsubfolder)

        # Loop over the .jpg files in the subsubfolder and append their names to the list
        for file_name in os.listdir(subsubfolder_path):
            if file_name.endswith(".jpg"):
                jpg_files.append(file_name)

############################################################################

import os
import random
import shutil

# Set the random seed to ensure reproducibility
random.seed(42)

# Set the paths to the source and destination directories
source_dir = "/home/giang/Downloads/RN50_dataset_CUB_HIGH/val/Correct"
dest_dir = "/home/giang/Downloads/Final_RN50_dataset_CUB_HP/clean_train"

# Get a list of the subfolders in the destination directory
subfolders = os.listdir(dest_dir)

abs_paths = []
all_files = []

for subfolder in subfolders:
    # Construct the full path to the source subdirectory
    source_subdir_path = os.path.join(source_dir, subfolder)

    # Get a list of the .jpg files in the source subdirectory
    files = [file for file in os.listdir(source_subdir_path) if file.endswith(".jpg")]
    for file in files:
        abs_paths.append(os.path.join(source_subdir_path, file))

    all_files.extend(files)
# # Sample 640 images from the list of .jpg files
# sampled_jpg_files = random.sample(jpg_files, min(len(jpg_files), 640))

random.seed(42)
random.shuffle(abs_paths)
cnt = 0
# Loop over the sampled .jpg files and copy them to the corresponding subdirectory in the destination directory
for idx, abs_path in enumerate(abs_paths):
    base_name = os.path.basename(abs_path)
    if base_name in jpg_files:
        continue
    cnt += 1
    # if cnt > 471:
    #     break

    bird_name = os.path.basename(os.path.dirname(abs_path))
    dest_subdir_path = os.path.join(dest_dir, bird_name)
    if not os.path.exists(dest_subdir_path):
        os.makedirs(dest_subdir_path)
    dest_file_path = os.path.join(dest_subdir_path, base_name)
    shutil.copyfile(abs_path, dest_file_path)

print(cnt)

