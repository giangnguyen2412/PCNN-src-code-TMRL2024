import os
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
import PIL
from tqdm import tqdm
import random
import collections
import torchvision.transforms as transforms
import glob
from IPython.core.debugger import Tracer
from PIL import Image
import shutil


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

print(key_list[200])


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


class GPUParams(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(device=0))

        self.device = torch.device("cuda:0")


# This block creates the Hard ImageNet dataset (possibly using ImageNetReaL labels)
import os
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
import PIL
from tqdm import tqdm
import random
import collections
import torchvision.transforms as transforms
import glob
import json
from IPython.core.debugger import Tracer
from PIL import Image

# This block is to get the hard/easy/medium distribution of ImageNet/Stanford Dogs
IMAGENET_REAL = False

model = torchvision.models.resnet18(pretrained=True).cuda()
model.eval()
hard = 0
easy = 0
medium = 0

from shutil import copyfile

imagenet_folders = glob.glob("/home/giang/Downloads/train/*")

dataset_path = "/home/giang/Downloads/RN18_dataset/"

check_and_mkdir(dataset_path)
check_and_mkdir("/home/giang/Downloads/RN18_dataset/Medium")
check_and_mkdir("/home/giang/Downloads/RN18_dataset/Easy")
check_and_mkdir("/home/giang/Downloads/RN18_dataset/Hard")
check_and_mkdir("/home/giang/Downloads/RN18_dataset/Correct")
check_and_mkdir("/home/giang/Downloads/RN18_dataset/Wrong")

if IMAGENET_REAL:
    real_json = open("../reassessed-imagenet/real.json")
    real_ids = json.load(real_json)
    real_labels = {
        f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels for i, labels in enumerate(real_ids)
    }

correct = 0
wrong = 0

RN18_dict = {'Correct': [], 'Wrong': []}

for i, imagenet_folder in enumerate(imagenet_folders):
    print(i)
    print([correct, wrong])
    imagenet_id = os.path.basename(imagenet_folder)
    wnid = imagenet_id

    image_paths = glob.glob(imagenet_folder + "/*.*")
    for idx, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        if img.mode != "RGB" or img.size[0] < 224 or img.size[1] < 224:
            continue

        img_name = os.path.basename(image_path)
        x = transform(img).unsqueeze(0).cuda()
        out = model(x)
        p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(p, 1)
        confidence_score = score[0][0].item()
        category_id = int(index[0][0].item())
        prediction_id = convert_imagenet_label_to_id(
            label_map, key_list, val_list, category_id
        )

        if IMAGENET_REAL:
            if category_id in real_labels[img_name]:
                correctness = True
            else:
                correctness = False
        else:
            if prediction_id == imagenet_id:
                correctness = True
            else:
                correctness = False

        #         print(prediction_id, imagenet_id)
        if correctness:
            correct += 1
            check_and_mkdir("/home/giang/Downloads/RN18_dataset/Correct/{}".format(wnid))
            dst_file = "/home/giang/Downloads/RN18_dataset/Correct/{}/{}".format(
                wnid, img_name
            )
        else:
            wrong += 1
            check_and_mkdir("/home/giang/Downloads/RN18_dataset/Wrong/{}".format(wnid))
            dst_file = "/home/giang/Downloads/RN18_dataset/Wrong/{}/{}".format(
                wnid, img_name
            )

        copyfile(image_path, dst_file)

print(medium, easy, hard)
print([correct, wrong])
