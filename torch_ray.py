# from torchray.benchmark import get_example_data, plot_example
# import torchvision
# import torch
# from PIL import Image
#
# # Pre-process the image and convert into a tensor
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(256),
#     torchvision.transforms.CenterCrop(224),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
# ])
#
# model = torchvision.models.resnet50(pretrained=True).eval().cuda()
# img = Image.open('/home/giang/tmp/Evening_Grosbeak_0118_37299.jpg')
# x = transform(img).unsqueeze(0).cuda()
#
# out = model(x)
# p = torch.nn.functional.softmax(out, dim=1)
# score, index = torch.topk(p, 1)
# category_id = index[0][0].item()
# predicted_confidence = score[0][0].item()
# predicted_confidence = "%.2f" % predicted_confidence
#
# print(predicted_confidence)
#
# # Grad-CAM backprop.
#
# # Plots.
# plot_example(x, saliency, 'grad-cam backprop', category_id)


import sys
from shutil import copyfile
import os
import numpy as np
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import torchvision.transforms as transforms
import wandb
import random
from PIL import Image
from operator import itemgetter
from torchray.attribution.grad_cam import grad_cam

from datasets import Dataset
from helper import HelperFunctions
from params import RunningParams
from visualize import Visualization
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
Visualization = Visualization()

imagenet_train_path = '/home/giang/Downloads/train/'
imagenet_train_data = ImageFolder(root=imagenet_train_path, transform=Dataset.imagenet_transform)

model = torchvision.models.resnet50(pretrained=True).eval().cuda()
layer = 4
K = 5
bs = 128
cam_RNlayer = 'layer4'

train_loader = torch.utils.data.DataLoader(
    imagenet_train_data,
    batch_size=bs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

for batch_idx, (data, gt) in enumerate(train_loader):
    x = data.cuda()
    gts = gt.cuda()
    out = model(x)
    p = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.topk(p, 1, dim=1)
    predicted_ids = index.squeeze()

    model2_gt = (predicted_ids == gts)*1
    explanation = grad_cam(model, x, index, saliency_layer=cam_RNlayer, resize=True)
    # tripe = (x, saliency, p, correctness?)
    pass



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--top1_only', action=argparse.BooleanOptionalAction,
#                         help='Generate explanations for top labels')
#     parser.add_argument('--clip', action=argparse.BooleanOptionalAction,
#                         help='Using CLIP features')
#
#     args = parser.parse_args()
#     print(args)
#     if args.top1_only:
#         HelperFunctions.check_and_rm('/home/giang/Downloads/posthoc-KNN/tmp/top1_only/')
#     else:
#         HelperFunctions.check_and_rm('/home/giang/Downloads/posthoc-KNN/tmp/no-top1_only/')

