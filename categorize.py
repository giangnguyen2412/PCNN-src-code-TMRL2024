import torchvision
import shutil
from tqdm import tqdm


# Added for loading ImageNet classes
def load_imagenet_label_map():
    input_f = open(f"{RunningParams.parent_dir}/kNN-classifiers/input_txt_files/imagenet_classes.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(": ")
        (num, label) = (int(parts[0]), parts[1].replace('"', ""))
        label_map[num] = label

    input_f.close()
    return label_map


# Added for loading ImageNet classes
def load_imagenet_id_map():
    input_f = open(f"{RunningParams.parent_dir}/kNN-classifiers/input_txt_files/synset_words.txt")
    label_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        (num, label) = (parts[0], " ".join(parts[1:]))
        label_map[num] = label

    input_f.close()
    return label_map


def load_imagenet_validation_gt():
    count = 0
    input_f = open(f"{RunningParams.parent_dir}/kNN-classifiers/input_txt_files/ILSVRC2012_validation_ground_truth.txt")
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


class GPUParams(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(device=0))

        self.device = torch.device("cuda:0")


# This block creates the Hard ImageNet dataset (possibly using ImageNetReaL labels)
import os
import torch
import torchvision
import glob
import json
from PIL import Image
import numpy as np
from shutil import copyfile

# This block is to get the hard/easy/medium distribution of ImageNet/Stanford Dogs
IMAGENET_REAL = False

model = torchvision.models.resnet34(pretrained=True).cuda()
model.eval()


imagenet_folders = glob.glob(f"{RunningParams.parent_dir}/val/*")

TRAIN_DOG = True
if TRAIN_DOG == True:
    imagenet_folders = glob.glob(f'{RunningParams.parent_dir}/SDogs_dataset/train/*')

    def load_imagenet_dog_label():
        count = 0
        dog_id_list = list()
        input_f = open(f"{RunningParams.parent_dir}/ImageNet_Dogs_dataset/dog_type.txt")
        for line in input_f:
            dog_id = (line.split('-')[0])
            dog_id_list.append(dog_id)
        return dog_id_list


    dogs_id = load_imagenet_dog_label()

dataset_path = f"{RunningParams.parent_dir}/RN34_SDogs_train"


check_and_rm(dataset_path)
check_and_mkdir(dataset_path)
check_and_mkdir("{}/Correct".format(dataset_path))
check_and_mkdir("{}/Wrong".format(dataset_path))


if IMAGENET_REAL:
    real_json = open("reassessed-imagenet/real.json")
    real_ids = json.load(real_json)
    real_labels = {
        f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels for i, labels in enumerate(real_ids)
    }

correct, wrong, easy, medium, hard = 0,0,0,0,0

RN_dict = {}

for i, imagenet_folder in enumerate(tqdm(imagenet_folders)):

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
        prediction_id = convert_imagenet_label_to_id(  # wnid
            label_map, key_list, val_list, category_id
        )

        if TRAIN_DOG:
            if prediction_id not in dogs_id:
                continue

        if IMAGENET_REAL is True:
            if len(real_labels[img_name]) == 0:
                continue

            if category_id in real_labels[img_name]:
                correctness = True
            else:
                correctness = False
        else:
            if prediction_id == imagenet_id:
                correctness = True
            else:
                correctness = False

        RN_dict[img_name] = {}
        RN_dict[img_name]['Output'] = correctness
        RN_dict[img_name]['confidence_score'] = confidence_score
        RN_dict[img_name]['predicted_category_id'] = category_id
        RN_dict[img_name]['predicted_wnid'] = prediction_id
        if IMAGENET_REAL is True:
            RN_dict[img_name]['gt_category_id'] = real_labels[img_name]
        else:
            RN_dict[img_name]['gt_wnid'] = imagenet_id

        if correctness is True:
            correct += 1
            output = 'Correct'

        else:
            wrong += 1
            output = 'Wrong'

        img_dir = os.path.join(dataset_path, output, wnid)
        check_and_mkdir(img_dir)
        dst_file = os.path.join(img_dir, img_name)

        copyfile(image_path, dst_file)

# np.save('RN18_train_dict', RN_dict)

print(medium, easy, hard)
print([correct, wrong])

