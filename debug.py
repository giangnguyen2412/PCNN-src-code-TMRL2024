import torch
import torch.nn as nn

# Create a tensor of size 200x2048
tensor = torch.randn(200, 2048)
print(tensor[0,:12])
tensor = tensor.unsqueeze(dim=1)
print(tensor.shape)

# Define the 1D average pooling layer
avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)

# Apply the 1D average pooling layer to the tensor
output = avg_pool(tensor).squeeze()
print(output[0,:4])


# The output tensor will have size 200x512
print(output.shape)  # prints "torch.Size([200, 512])"


exit(0)

import torchvision
import torch
from collections import OrderedDict
import glob
import os

images = glob.glob('/home/giang/Downloads/advising_network/mixed_images/*.*')
dog_images = []
for image in images:
    print(image)
    base_name = os.path.basename(image)
    tokens = base_name.split('_')
    dog_image = '_'.join(tokens[1:-1]) + '.JPEG'
    dog_images.append(dog_image)
pass


import numpy as np
from datasets import ImageFolderForNNs
from helpers import HelperFunctions
import os
import glob

filename = 'faiss/faiss_SDogs_val_RN34_top1.npy'
kbc = np.load(filename, allow_pickle=True, ).item()

HelperFunctions = HelperFunctions()

def visualize_dog_prototypes(
        query: str,
        gt_label: str,
        prototypes: list,
        save_dir: str):
    cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
        query, save_dir
    )
    os.system(cmd)
    annotation = gt_label
    cmd = 'convert {}/query.jpeg -font aakar -pointsize 10 -gravity North -background ' \
          'White -splice 0x40 -annotate +0+4 "{}" {}/query.jpeg'.format(
        save_dir, annotation, save_dir
    )
    os.system(cmd)
    for idx, prototype in enumerate(prototypes):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
            prototype, save_dir, idx)
        os.system(cmd)
        annotation = prototype.split('/')[-2]

        annotation = HelperFunctions.convert_imagenet_id_to_label(HelperFunctions.key_list, annotation)
        annotation = HelperFunctions.label_map[annotation]

        cmd = 'convert {}/{}.jpeg -font aakar -pointsize 10 -gravity North -background ' \
              'White -splice 0x40 -annotate +0+4 "{}" {}/{}.jpeg'.format(
            save_dir, idx, annotation, save_dir, idx
        )
        os.system(cmd)

    cmd = 'montage {}/[0-5].jpeg -tile 6x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
    os.system(cmd)

    file_name = os.path.basename(query)
    cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}/{}.JPEG'.format(save_dir, save_dir,
                                                                                               save_dir, file_name)
    os.system(cmd)

id = HelperFunctions.load_imagenet_validation_gt()
cnt = 0

for query, nn in kbc.items():
    cnt += 1
    if cnt == 10:
        break

    if 'train' in filename:
        wnid = query.split('_')[0]
        query = os.path.join('/home/giang/Downloads/datasets/Dogs_train', wnid, query)
    else:
        path = glob.glob('/home/giang/Downloads/datasets/imagenet1k-val/**/{}'.format(query))
        wnid = path[0].split('/')[-2]
        query = os.path.join('/home/giang/Downloads/datasets/Dogs_val', wnid, query)

    gt_label = HelperFunctions.convert_imagenet_id_to_label(HelperFunctions.key_list, wnid)
    gt_label = HelperFunctions.label_map[gt_label]
    visualize_dog_prototypes(query, gt_label, nn, 'tmp')


