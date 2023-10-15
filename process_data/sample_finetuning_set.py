import os
import random
import shutil
from params import RunningParams

RunningParams = RunningParams()
set = 'test'
source_folder = f"{RunningParams.parent_dir}/RN50_dataset_CUB_HIGH/val/Wrong"
destination_folder = "{}/RN50_dataset_CUB_Finetuning/{}".format(RunningParams.parent_dir, set)
if 'train' == set:
    num_files = 500
elif 'val' == set:
    num_files = 160
elif 'test' == set:
    num_files = 160

# TODO: train 500, val 160, test 160

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through subfolders in the source folder
for dirpath, dirnames, filenames in os.walk(source_folder):
    # Determine the corresponding subfolder in the destination folder
    relative_path = os.path.relpath(dirpath, source_folder)
    dest_dir = os.path.join(destination_folder, relative_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Randomly choose up to num_files files to copy from this subfolder
    files_to_copy = random.sample(filenames, min(num_files, len(filenames)))

    # Copy the selected files to the corresponding subfolder in the destination folder
    for filename in files_to_copy:
        source_file = os.path.join(dirpath, filename)
        dest_file = os.path.join(dest_dir, filename)
        # shutil.copy2(source_file, dest_file)
        shutil.move(source_file, dest_file)

        num_files -= 1
        if num_files == 0:
            break

    if num_files == 0:
        break

print("Random files copied to: ", destination_folder)
