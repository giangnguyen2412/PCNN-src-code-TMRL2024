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


class GPUParams(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(device=0))

        self.device = torch.device("cuda:4")


# This block creates the Hard ImageNet dataset (possibly using ImageNetReaL labels)
import os
import torch
import torchvision
import glob
import json
from PIL import Image
import numpy as np
from shutil import copyfile

device = torch.device("cuda:4")


# This block is to get the hard/easy/medium distribution of ImageNet/Stanford Dogs
IMAGENET_REAL = False

model = torchvision.models.resnet34(pretrained=True).to(device)

CUB = True
if CUB is True:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        '/home/giang/Downloads/visual-correspondence-XAI/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    print("LOADED.......")
    model.load_state_dict(my_model_state_dict, strict=True)
    model.to(device)

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    validation_folder = ImageFolder(root='/home/giang/Downloads/datasets/CUB/combined/',
                                    transform=transform)
    val_loader = DataLoader(validation_folder, batch_size=512, shuffle=False, num_workers=8, pin_memory=False)

imagenet_folders = glob.glob("/home/giang/Downloads/train/*")

if CUB is True:
    imagenet_folders = glob.glob("/home/giang/Downloads/datasets/CUB/combined/*")

TRAIN_DOG = False
if TRAIN_DOG == True:
    imagenet_folders = glob.glob('/home/giang/Downloads/SDogs_dataset/val/*')
    # imagenet_folders = '/home/giang/Downloads/Dogs_dataset/val/*'

dataset_path = "/home/giang/Downloads/RN50_dataset_CUB"

check_and_rm(dataset_path)
check_and_mkdir(dataset_path)
if not CUB:
    check_and_mkdir("{}/Correct".format(dataset_path))
    check_and_mkdir("{}/Correct/Medium".format(dataset_path))
    check_and_mkdir("{}/Correct/Easy".format(dataset_path))
    check_and_mkdir("{}/Correct/Hard".format(dataset_path))
    check_and_mkdir("{}/Wrong".format(dataset_path))
    check_and_mkdir("{}/Wrong/Medium".format(dataset_path))
    check_and_mkdir("{}/Wrong/Easy".format(dataset_path))
    check_and_mkdir("{}/Wrong/Hard".format(dataset_path))
if CUB:
    check_and_mkdir("{}/True".format(dataset_path))
    check_and_mkdir("{}/False".format(dataset_path))

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
        if not CUB:
            if img.mode != "RGB" or img.size[0] < 224 or img.size[1] < 224:
                continue
        if CUB:
            if img.mode != "RGB":
                continue

        img_name = os.path.basename(image_path)
        x = transform(img).unsqueeze(0).to(device)
        out = model(x)
        if CUB:
            p = torch.nn.functional.softmax(out.unsqueeze(0), dim=1)
        else:
            p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(p, 1)
        confidence_score = score[0][0].item()
        category_id = int(index[0][0].item())
        prediction_id = convert_imagenet_label_to_id(
            label_map, key_list, val_list, category_id
        )

        if IMAGENET_REAL is True:
            if len(real_labels[img_name]) == 0:
                continue

            if category_id in real_labels[img_name]:
                correctness = True
            else:
                correctness = False
        else:
            if CUB:
                prediction_id = val_loader.dataset.classes[category_id]

            if prediction_id == imagenet_id:
                correctness = True
                correct += 1
            else:
                correctness = False
                wrong += 1

        RN_dict[img_name] = {}
        RN_dict[img_name]['Output'] = correctness
        RN_dict[img_name]['confidence_score'] = confidence_score
        RN_dict[img_name]['predicted_category_id'] = category_id
        RN_dict[img_name]['predicted_wnid'] = prediction_id
        if IMAGENET_REAL is True:
            RN_dict[img_name]['gt_category_id'] = real_labels[img_name]
        else:
            RN_dict[img_name]['gt_wnid'] = imagenet_id

        if not CUB:
            if correctness is True:
                correct += 1
                output = 'Correct'
                if confidence_score < 0.3:
                    harness = 'Hard'
                    hard += 1
                elif 0.4 <= confidence_score < 0.6:
                    harness = 'Medium'
                    medium += 1
                elif confidence_score >= 0.8:
                    harness = 'Easy'
                    easy += 1
                else:
                    continue

            else:
                wrong += 1
                output = 'Wrong'
                if confidence_score < 0.3:
                    harness = 'Easy'
                    easy += 1
                elif 0.4 <= confidence_score < 0.6:
                    harness = 'Medium'
                    medium += 1
                elif confidence_score >= 0.8:
                    harness = 'Hard'
                    hard += 1
                else:
                    continue

        #TODO : consider change this
        # img_dir = os.path.join(dataset_path, output, harness, wnid)
        img_dir = os.path.join(dataset_path, str(correctness), wnid)
        check_and_mkdir(img_dir)
        dst_file = os.path.join(img_dir, img_name)

        # copyfile(image_path, dst_file)

# np.save('RN_train_dict', RN_dict)

print(medium, easy, hard)
print([correct, wrong])
