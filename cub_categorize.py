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

# import glob
# test_path = '/home/giang/Downloads/RN50_dataset_CUB_Pretraining/val'
# test_samples = []
# image_folders = glob.glob('{}/*'.format(test_path))
# for i, image_folder in enumerate(image_folders):
#     id = image_folder.split('val/')[1]
#     image_paths = glob.glob(image_folder + '/*.*')
#     for image in image_paths:
#         test_samples.append(os.path.basename(image))



concat = lambda x: np.concatenate(x, axis=0)
to_np  = lambda x: x.data.to('cpu').numpy()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

val_dataset_transform = transforms.Compose(
  [transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_folder = ImageFolderWithPaths(root='/home/giang/Downloads/datasets/CUB/train1', transform=val_dataset_transform)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=True, num_workers=8, pin_memory=False)

from params import RunningParams
RunningParams = RunningParams()

HIGHPERFORMANCE_FEATURE_EXTRACTOR = RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR
HIGHPERFORMANCE_FEATURE_EXTRACTOR = False
if HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    inat_resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'pretrained_models/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    inat_resnet.load_state_dict(my_model_state_dict, strict=True)
    # to CUDA
    inat_resnet.cuda()
    inat_resnet.eval()
else:
    import torchvision

    inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
    inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
    my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
    inat_resnet.load_state_dict(my_model_state_dict, strict=True)
    # Freeze backbone (for training only)
    # to CUDA
    inat_resnet.cuda()
    inat_resnet.eval()

dataset_path = "/home/giang/Downloads/RN50_dataset_CUB_LOW"

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

    running_corrects = 0

    train_cnt, test_cnt = 0, 0
    train_crt, train_wrong, test_crt, test_wrong = 0, 0, 0, 0
    overlap = 0

    predictions = []
    confidence = []

    with torch.inference_mode():
        for batch_idx, (data, target, pts) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
            running_corrects += torch.sum(preds == target.data)

            predictions.extend(preds.data.cpu().numpy())
            confidence.extend((probs.data.cpu().numpy() * 100).astype(np.int32))

            for sample_idx in range(data.shape[0]):
                base_name = os.path.basename(pts[sample_idx])
                wnid = val_loader.dataset.classes[target[sample_idx]]

                if 'train' in pts[sample_idx]:
                    if preds[sample_idx] == target[sample_idx]:
                        check_and_mkdir(os.path.join(correct_train_path, wnid))
                        dst_dir = os.path.join(correct_train_path, wnid, base_name)
                        train_crt += 1
                    else:
                        check_and_mkdir(os.path.join(wrong_train_path, wnid))
                        dst_dir = os.path.join(wrong_train_path, wnid, base_name)
                        train_wrong += 1
                else:
                    if preds[sample_idx] == target[sample_idx]:
                        check_and_mkdir(os.path.join(correct_test_path, wnid))
                        dst_dir = os.path.join(correct_test_path, wnid, base_name)
                        test_crt += 1
                    else:
                        check_and_mkdir(os.path.join(wrong_test_path, wnid))
                        dst_dir = os.path.join(wrong_test_path, wnid, base_name)
                        test_wrong += 1

                # shutil.copyfile(pts[sample_idx], dst_dir)

    epoch_acc = running_corrects.double() / len(validation_folder)

    print('-' * 10)
    print('Acc: {:.4f}'.format(100 * epoch_acc))
    return predictions, confidence, (train_crt, train_wrong, test_crt, test_wrong)


cub_test_preds, cub_test_confs, counts = test_cub(inat_resnet)
print('Train Correct: {} - Train Wrong: {} - Test Correct: {} - Test Wrong: {}'.format(counts[0], counts[1], counts[2], counts[3]))

