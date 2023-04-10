import os
import numpy as np

# Define the path to the folder
folder_path = "/home/giang/Downloads/RN50_dataset_CUB_Finetuning"

# Create an empty dictionary to store the results
file_dict = {}

# Loop through the subfolders in the main folder
for subdir in os.listdir(folder_path):
    # Ignore any non-folders
    if not os.path.isdir(os.path.join(folder_path, subdir)):
        continue
    # Create an empty list to store the files in the subfolders
    subfolder_files = []
    # Loop through the subfolders in the current subfolder
    for subsubdir in os.listdir(os.path.join(folder_path, subdir)):
        # Ignore any non-folders
        if not os.path.isdir(os.path.join(folder_path, subdir, subsubdir)):
            continue
        # Get a list of the files in the sub-subfolder
        files = os.listdir(os.path.join(folder_path, subdir, subsubdir))
        # Add the files to the list of files in the current subfolder
        subfolder_files += files
    # Add the list of files to the dictionary under the subfolder name
    file_dict[subdir] = subfolder_files

np.save('../Finetuning.npy', file_dict)
# Print the resulting dictionary
print(file_dict)