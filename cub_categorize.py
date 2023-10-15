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
import torchvision.transforms as transforms

from datasets import ImageFolderWithPaths

import torchvision
import shutil

def check_and_mkdir(f):
    if not os.path.exists(f):
        os.mkdir(f)
    else:
        pass


def check_and_rm(f):
    if os.path.exists(f):
        shutil.rmtree(f)
    else:
        pass


# Pre-process the image and convert into a tensor
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


concat = lambda x: np.concatenate(x, axis=0)
to_np  = lambda x: x.data.to('cpu').numpy()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

val_dataset_transform = transforms.Compose(
  [transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_folder = ImageFolderWithPaths(root=f'{RunningParams.parent_dir}/datasets/CUB/test0', transform=val_dataset_transform)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=False)

from params import RunningParams
RunningParams = RunningParams()

from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

inat_resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
inat_resnet.load_state_dict(my_model_state_dict, strict=True)
# to CUDA
inat_resnet.cuda()
inat_resnet.eval()

dataset_path = f"{RunningParams.parent_dir}/RN50_dataset_CUB_LOW"

# check_and_rm(dataset_path)
check_and_mkdir(dataset_path)
correct, wrong = 0, 0

train_path = "{}/train".format(dataset_path)
test_path = "{}/val".format(dataset_path)
correct_train_path = "{}/train/Correct".format(dataset_path)
wrong_train_path = "{}/train/Wrong".format(dataset_path)
correct_test_path = "{}/val/Correct".format(dataset_path)
wrong_test_path = "{}/val/Wrong".format(dataset_path)

check_and_mkdir(train_path)
check_and_mkdir(test_path)
check_and_mkdir(correct_train_path)
check_and_mkdir(wrong_train_path)
check_and_mkdir(correct_test_path)
check_and_mkdir(wrong_test_path)


def test_cub(model):
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

test_cub(inat_resnet)