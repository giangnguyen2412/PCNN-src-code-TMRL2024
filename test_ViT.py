import os
import torch
import torchvision.transforms as transforms
import torchvision.io
from PIL import Image
import numpy as np


folder_path = "/home/giang/Downloads/RN50_dataset_CUB_LP/train"

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create an empty dictionary
file_dict = {}

# Loop through all the subfolders in the main folder
for subdir in os.listdir(folder_path):
    # Get the full path of the subfolder
    subdir_path = os.path.join(folder_path, subdir)

    # Create an empty list for this subfolder's tensors
    tensor_list = []

    # Loop through the files in the subfolder
    for file_name in os.listdir(subdir_path):
        # Get the full path of the file
        file_path = os.path.join(subdir_path, file_name)

        img = Image.open(file_path)
        if img.mode != "RGB" or img.size[0] < 224 or img.size[1] < 224:
            continue

        x = transform(img).squeeze().cuda()

        # Add the tensor to the list
        tensor_list.append(x)

        # Stop after adding 3 tensors
        if len(tensor_list) == 3:
            break

    # Stack the list of tensors into a single tensor
    stacked_tensor = torch.stack(tensor_list)

    # Add the subfolder and its stacked tensor to the dictionary
    file_dict[subdir] = stacked_tensor

np.save('file_dict.npy', file_dict)

