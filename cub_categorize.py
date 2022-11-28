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
from tqdm import tqdm


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

from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
concat = lambda x: np.concatenate(x, axis=0)
to_np  = lambda x: x.data.to('cpu').numpy()
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
val_dataset_transform = transforms.Compose(
  [transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_folder = ImageFolderWithPaths(root='/home/giang/Downloads/datasets/CUB/combined/', transform=val_dataset_transform)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=False)



import glob
test_path = '/home/giang/Downloads/RN50_dataset_CUB_LP/val'
train_path = '/home/giang/Downloads/RN50_dataset_CUB_LP/train'
check_and_rm(train_path)
check_and_mkdir(train_path)


# Sample test images
check_and_rm(test_path)
cmd = 'sh random_sample_dataset.sh -d /home/giang/Downloads/datasets/CUB/test0 -s 5 -o {}'.format(test_path)
os.system(cmd)

test_samples = []
image_folders = glob.glob('{}/*'.format(test_path))
for i, image_folder in enumerate(image_folders):
    id = image_folder.split('val/')[1]
    image_paths = glob.glob(image_folder + '/*.*')
    for image in image_paths:
        test_samples.append(os.path.basename(image))

from params import RunningParams
RunningParams = RunningParams()

HIGHPERFORMANCE_MODEL1 = RunningParams.HIGHPERFORMANCE_MODEL1
if HIGHPERFORMANCE_MODEL1 is True:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    inat_resnet = resnet
else:
    import torchvision
    inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
    inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
    my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
    inat_resnet.load_state_dict(my_model_state_dict, strict=True)

# Freeze backbone (for training only)
for param in list(inat_resnet.parameters())[:-2]:
    param.requires_grad = False

# to CUDA
inat_resnet.to(device)

criterion = nn.CrossEntropyLoss()

correct, wrong = 0, 0


def test_cub(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    train_cnt, test_cnt = 0, 0

    predictions = []
    confidence = []

    with torch.inference_mode():
        for batch_idx, (data, target, pts) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)
            probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
            running_loss += loss.item() * target.size(0)
            running_corrects += torch.sum(preds == target.data)

            predictions.extend(preds.data.cpu().numpy())
            confidence.extend((probs.data.cpu().numpy() * 100).astype(np.int32))

            for sample_idx in range(data.shape[0]):
                base_name = os.path.basename(pts[sample_idx])
                wnid = val_loader.dataset.classes[target[sample_idx]]

                if base_name not in test_samples:
                    check_and_mkdir(os.path.join(train_path, wnid))
                    dst_dir = os.path.join(train_path, wnid, base_name)
                else:
                    continue
                shutil.copyfile(pts[sample_idx], dst_dir)

    epoch_loss = running_loss / len(validation_folder)
    epoch_acc = running_corrects.double() / len(validation_folder)

    print('-' * 10)
    print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100 * epoch_acc))

    return predictions, confidence

cub_test_preds, cub_test_confs = test_cub(inat_resnet)