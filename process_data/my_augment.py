import os
import torchvision.transforms as transforms
from PIL import Image

ULTIMATE_TRANSFORM = True

# Define the transformations you want to apply
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
])

transform2 = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
])

transform3 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
])

transform4 = transforms.Compose([
    transforms.RandomResizedCrop(224),
])


transform5 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomResizedCrop(224),
])

if ULTIMATE_TRANSFORM is True:
    ultimate_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.25, 0.25)),
    ])


transforms_list = [transform1, transform2, transform3, transform4, transform5]


# Set the path to your dataset
data_path = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HP/tmp_train_selfaugmentv2'

# Loop over each subfolder in the dataset
for subfolder_name in os.listdir(data_path):
    subfolder_path = os.path.join(data_path, subfolder_name)

    # Loop over each image in the subfolder
    for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        if 'jpg' not in image_path:
            continue

        # Open the image
        image = Image.open(image_path)

        # Apply each transformation to the image and save the transformed image back to
        for idx in range(5):
            if ULTIMATE_TRANSFORM is True:
                augmented_img = ultimate_transform(image)
            else:
                augmented_img = transforms_list[idx](image)

            # Save the transformed image back to its original location
            new_image_path = os.path.join(subfolder_path, "{}_aug_{}".format(idx, image_name))
            augmented_img.save(new_image_path)
