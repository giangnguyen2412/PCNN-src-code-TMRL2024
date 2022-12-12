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


# Added for loading ImageNet classes
def load_imagenet_label_map():
    input_f = open("/home/giang/Downloads/kNN-classifiers/input_txt_files/imagenet_classes.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(": ")
        (num, label) = (int(parts[0]), parts[1].replace('"', ""))
        label_map[num] = label

    input_f.close()
    return label_map


# Added for loading ImageNet classes
def load_imagenet_id_map():
    input_f = open("/home/giang/Downloads/kNN-classifiers/input_txt_files/synset_words.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        (num, label) = (parts[0], " ".join(parts[1:]))
        label_map[num] = label

    input_f.close()
    return label_map


def load_imagenet_validation_gt():
    count = 0
    input_f = open("/home/giang/Downloads/kNN-classifiers/input_txt_files/ILSVRC2012_validation_ground_truth.txt")
    gt_dict = {}
    while True:
        count += 1

        # Get the next line
        line = input_f.readline()

        # if line is empty, EOL is reached
        if not line:
            break
        gt_dict[count] = int(line.strip())

    input_f.close()
    return gt_dict


def convert_imagenet_label_to_id(label_map, key_list, val_list, prediction_class):
    class_to_label = label_map[prediction_class]
    prediction_id = key_list[val_list.index(class_to_label)]
    return prediction_id


gt_dict = load_imagenet_validation_gt()
id_map = load_imagenet_id_map()
label_map = load_imagenet_label_map()

key_list = list(id_map.keys())
val_list = list(id_map.values())


def convert_imagenet_id_to_label(key_list, class_id):
    return key_list.index(str(class_id))


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

import glob
test_path = '/home/giang/Downloads/RN50_dataset_CUB_LP/val'
test_samples = []
image_folders = glob.glob('{}/*'.format(test_path))
for i, image_folder in enumerate(image_folders):
    id = image_folder.split('val/')[1]
    image_paths = glob.glob(image_folder + '/*.*')
    for image in image_paths:
        test_samples.append(os.path.basename(image))


from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
concat = lambda x: np.concatenate(x, axis=0)
to_np  = lambda x: x.data.to('cpu').numpy()
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
val_dataset_transform = transforms.Compose(
  [transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_folder = ImageFolderWithPaths(root='/home/giang/Downloads/datasets/CUB/test0', transform=val_dataset_transform)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=False)

from params import RunningParams
RunningParams = RunningParams()

HIGHPERFORMANCE_FEATURE_EXTRACTOR = RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR
if HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    # Freeze backbone (for training only)
    for param in list(resnet.parameters())[:-2]:
        param.requires_grad = False
    # to CUDA
    inat_resnet = resnet.cuda()
    inat_resnet.cuda()
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
    inat_resnet.cuda()


# Freeze backbone (for training only)
for param in list(inat_resnet.parameters())[:-2]:
    param.requires_grad = False

# to CUDA
inat_resnet.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(inat_resnet.classifier.parameters())

dataset_path = "/home/giang/Downloads/RN50_dataset_CUB_HP/"

check_and_rm(dataset_path)
check_and_mkdir(dataset_path)
correct, wrong = 0, 0

train_path = "{}/train".format(dataset_path)
test_path = "{}/val".format(dataset_path)
# correct_train_path = "{}/train/Correct".format(dataset_path)
# wrong_train_path = "{}/train/Wrong".format(dataset_path)
# correct_test_path = "{}/test/Correct".format(dataset_path)
# wrong_test_path = "{}/test/Wrong".format(dataset_path)

check_and_mkdir(train_path)
check_and_mkdir(test_path)
# check_and_mkdir(correct_train_path)
# check_and_mkdir(wrong_train_path)
# check_and_mkdir(correct_test_path)
# check_and_mkdir(wrong_test_path)


def test_cub(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    train_cnt, test_cnt = 0, 0
    overlap = 0

    predictions = []
    confidence = []

    with torch.inference_mode():
        for batch_idx, (data, target, pts) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()
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

                # if train_cnt < 10500:
                if 'train' in pts[sample_idx]:
                    if preds[sample_idx] == target[sample_idx]:
                        check_and_mkdir(os.path.join(correct_train_path, wnid))
                        dst_dir = os.path.join(correct_train_path, wnid, base_name)
                    else:
                        check_and_mkdir(os.path.join(wrong_train_path, wnid))
                        dst_dir = os.path.join(wrong_train_path, wnid, base_name)

                    train_cnt += 1
                else:
                    if preds[sample_idx] == target[sample_idx]:
                        # check_and_mkdir(os.path.join(correct_test_path, wnid))
                        # dst_dir = os.path.join(correct_test_path, wnid, base_name)
                        if base_name in test_samples and test_cnt < 145:
                            check_and_mkdir(os.path.join(test_path, wnid))
                            dst_dir = os.path.join(test_path, wnid, base_name)
                            test_cnt += 1
                        elif base_name not in test_samples:
                            if train_cnt < 675:
                                check_and_mkdir(os.path.join(train_path, wnid))
                                dst_dir = os.path.join(train_path, wnid, base_name)
                                train_cnt += 1
                    else:
                        if base_name in test_samples:
                            check_and_mkdir(os.path.join(test_path, wnid))
                            dst_dir = os.path.join(test_path, wnid, base_name)
                        elif base_name not in test_samples:
                            check_and_mkdir(os.path.join(train_path, wnid))
                            dst_dir = os.path.join(train_path, wnid, base_name)
                        # check_and_mkdir(os.path.join(wrong_test_path, wnid))
                        # dst_dir = os.path.join(wrong_test_path, wnid, base_name)
                shutil.copyfile(pts[sample_idx], dst_dir)

    epoch_loss = running_loss / len(validation_folder)
    epoch_acc = running_corrects.double() / len(validation_folder)

    print('-' * 10)
    print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100 * epoch_acc))
    print(overlap)
    return predictions, confidence

cub_test_preds, cub_test_confs = test_cub(inat_resnet)