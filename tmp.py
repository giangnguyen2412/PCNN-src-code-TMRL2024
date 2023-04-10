from datasets import ImageFolderForNNs
from helpers import HelperFunctions
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torchvision

# resnet = torchvision.models.resnet34(pretrained=True).cuda()

HelperFunctions = HelperFunctions()


filename = 'confidence.npy'
confidence_dict = np.load(filename, allow_pickle=True, ).item()

confidence_dict = dict(sorted(confidence_dict.items()))

def calculate_accuracy(confidence_dict, key):
    correct_count = 0
    total_count = 0
    for confidence, counts in confidence_dict.items():
        if confidence >= key:
            correct_count += counts[0]
            total_count += sum(counts)
    accuracy = correct_count*100/total_count
    return accuracy, total_count


# conf < T -> advising network | conf >= T -> thresholding
def calculate_accuracy_v2(confidence_dict, key):
    correct_count = 0
    total_count = 0
    for confidence, counts in confidence_dict.items():
        if confidence < key:
            correct_count += counts[0]
        else:
            correct_count += counts[2]

        total_count += counts[0] + counts[1]

    accuracy = correct_count*100/279
    return accuracy, total_count

# Example usage
for confidence, counts in confidence_dict.items():
    accuracy, total_count = calculate_accuracy_v2(confidence_dict, confidence)
    if confidence %5 == 0 or confidence == 99:
        # print(f"Confidence scores: >= {confidence}, Advising network: {accuracy:.2f}, % of samples: {total_count*100/279:.2f}")
        print(f"Confidence scores: >= {confidence}, Advising network: {accuracy:.2f}")



# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Open the image
# img = Image.open('tmp.jpg')
#
#
# # Get the original dimensions of the image
# width, height = img.size
#
# # Calculate the amount of padding needed
# diff = abs(width - height)
# padding = (0, diff//2, 0, diff//2) if width > height else (diff//2, 0, diff//2, 0)
#
# # Define the transform to pad and resize the image
# padding_transform = transforms.Compose([
#     transforms.Pad(padding, fill=0, padding_mode='constant'), # Pad the image
#     transforms.Resize((224, 224), interpolation=Image.BILINEAR), # Resize to 224x224 using bilinear interpolation
#     transforms.ToTensor() # Convert the image to a PyTorch tensor
# ])
#
# # Define the transform to resize the image
# resize_aspect = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=Image.BILINEAR), # Resize to 224x224 using bilinear interpolation
#     transforms.ToTensor() # Convert the image to a PyTorch tensor
# ])
#
# resize_center_crop = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#             ])
#
# # Define the transform to resize the image
# resize_aspect_center_crop = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.CenterCrop(224),
#     transforms.ToTensor() # Convert the image to a PyTorch tensor
# ])
#
#
#
# # Plot the original image and the transformed image side by side
# fig, axes = plt.subplots(1, 4)
# axes[0].imshow(img)
# axes[0].set_title('Original')
#
#
# # Apply the transform to the image
# img_tensor = resize_aspect(img)
# # Convert the tensor back to an image
# img_transformed = transforms.ToPILImage()(img_tensor)
# axes[1].imshow(img_transformed)
# axes[1].set_title('R bilinear 224')
#
# # Apply the transform to the image
# img_tensor = resize_center_crop(img)
# # Convert the tensor back to an image
# img_transformed = transforms.ToPILImage()(img_tensor)
# axes[2].imshow(img_transformed)
# axes[2].set_title('R256 CR 224')
#
# # Apply the transform to the image
# img_tensor = padding_transform(img)
# # Convert the tensor back to an image
# img_transformed = transforms.ToPILImage()(img_tensor)
# axes[3].imshow(img_transformed)
# axes[3].set_title('Padding -> resize')
# plt.show()



# filename = 'faiss/faiss_SDogs_val_check_RN34_top1.npy'
# file_a = np.load(filename, allow_pickle=True, ).item()
# cnt = 0
# for key, value in file_a.items():
#     cnt += 1
#     print(key, value[0])
#     img_list = [key] + value[0:3]
#     titles = []
#     for img in img_list:
#         dir = os.path.dirname(img)
#         titles.append(HelperFunctions.id_map[os.path.basename(dir)].split(',')[0])
#     ################################################################
#     # Create a figure with 1 row and 4 columns
#     fig, axs = plt.subplots(1, 4, figsize=(15, 5))
#
#     # Loop through the image paths and plot each image
#     for i, (img_path, title) in enumerate(zip(img_list, titles)):
#         img = mpimg.imread(img_path)
#         axs[i].imshow(img)
#         axs[i].set_title(title)
#         axs[i].axis('off')
#
#     # Show the figure
#     plt.show()
#     ################################################################




