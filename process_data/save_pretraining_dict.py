import numpy as np
import os

# Define the path to the Pretraining folder
pretraining_path = '/home/giang/Downloads/datasets/CUB/Pretraining'

# Define the train and val subfolder names
train_folder = 'train'
val_folder = 'val'

# Create an empty dictionary to hold the file names
file_dict = {'train': [], 'val': []}

# Loop over the train and val folders
for folder_name in [train_folder, val_folder]:

    # Get the path to the current folder
    folder_path = os.path.join(pretraining_path, folder_name)

    # Loop over the subfolders in the current folder
    for subfolder_name in os.listdir(folder_path):

        # Get the path to the current subfolder
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Loop over the jpg files in the current subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.jpg'):
                # Add the file name to the corresponding list in the dictionary
                file_dict[folder_name].append(os.path.join(subfolder_name, file_name))

np.save('Pretraining.npy', file_dict)
# Print the dictionary
print(len(file_dict['train']))
print(len(file_dict['val']))