# Compute the topk accuracy of a pretrained CUB classifier
import os.path
import random
random.seed(43)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from params import RunningParams
import torchvision.transforms as transforms

from datasets import ImageFolderWithPaths

import torchvision
import shutil

RunningParams = RunningParams()


concat = lambda x: np.concatenate(x, axis=0)
to_np  = lambda x: x.data.to('cpu').numpy()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

val_dataset_transform = transforms.Compose(
  [transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_folder = ImageFolderWithPaths(root=f'{RunningParams.parent_dir}/RunningParams.test_path', transform=val_dataset_transform)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=False)

from params import RunningParams
RunningParams = RunningParams()

from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

inat_resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    '/home/giang/Downloads/advising_network/pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
inat_resnet.load_state_dict(my_model_state_dict, strict=True)
# to CUDA
inat_resnet.cuda()
inat_resnet.eval()


def compute_topk_acc(model):
    model.eval()

    # Initialize variables to store top-k accuracies
    topk_corrects = {k: 0 for k in range(1, 11)}

    predictions = []
    confidence = []

    with torch.inference_mode():
        for batch_idx, (data, target, pts) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()
            outputs = model(data)

            # Calculate top-10 class indices for each image
            topk_indices = torch.topk(outputs, k=10, dim=1)[1]

            for k in range(1, 11):
                topk_corrects[k] += torch.sum(topk_indices[:, :k] == target.view(-1, 1)).cpu().item()

            _, preds = torch.max(outputs, 1)
            probs, _ = torch.max(F.softmax(outputs, dim=1), 1)

            predictions.extend(preds.data.cpu().numpy())
            confidence.extend((probs.data.cpu().numpy() * 100).astype(np.int32))

    total_samples = len(validation_folder)
    print('-' * 10)
    for k in range(1, 11):
        topk_acc = 100.0 * topk_corrects[k] / total_samples
        print(f'Top-{k} Acc: {topk_acc:.4f}')

compute_topk_acc(inat_resnet)