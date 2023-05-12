import os
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as T
import torch

augmenter = T.TrivialAugmentWide()
# Set the path to your dataset
data_path = '/home/giang/Downloads/datasets/CUB_train_aug'

# Loop over each subfolder in the dataset
for subfolder_name in os.listdir(data_path):
    subfolder_path = os.path.join(data_path, subfolder_name)

    # Loop over each image in the subfolder
    for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        if 'jpg' not in image_path:
        # if 'JPEG' not in image_path:
            continue

        # Open the image
        image = Image.open(image_path)

        # Apply each transformation to the image and save the transformed image back to
        for idx in range(5):
            augmented_img = augmenter(image)
            # Save the transformed image back to its original location
            new_image_path = os.path.join(subfolder_path, "{}_aug_{}".format(idx, image_name))
            # transformed_image = transforms.ToPILImage()(augmented_img)  # Convert back to PIL image
            augmented_img.save(new_image_path)
