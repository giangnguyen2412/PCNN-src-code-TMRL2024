import glob
import torch
import numpy as np

from torchvision.models import resnet18, resnet34, resnet50
from PIL import Image
from tqdm import tqdm
import torchvision

from params import RunningParams
from datasets import Dataset
from helpers import HelperFunctions

HelperFunctions = HelperFunctions()

T_list = list(np.arange(0.0, 1.0, 0.05))
imagenet_folders = glob.glob('/home/giang/Downloads/datasets/imagenet1k-val/*')

model = resnet18(pretrained=True).cuda()
model.eval()

# Pre-process the image and convert into a tensor
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

confidence_dict = dict()

for i, imagenet_folder in enumerate(tqdm(imagenet_folders)):
    imagenet_id = imagenet_folder.split('val/')[1]
    image_paths = glob.glob(imagenet_folder + '/*.*')

    for idx, image_path in enumerate(image_paths):
        img = Image.open(image_path)

        if img.mode != 'RGB':
            continue

        x = transform(img).unsqueeze(0).cuda()
        out = model(x)
        p = torch.nn.functional.softmax(out, dim=1)
        score, index = torch.topk(p, 1)
        confidence_score = score[0][0].item()
        category_id = int(index[0][0].item())
        prediction_id = HelperFunctions.convert_imagenet_label_to_id(HelperFunctions.label_map,
                                                                     HelperFunctions.key_list,
                                                                     HelperFunctions.val_list,
                                                                     category_id)
        confidence_dict[image_path] = dict()
        confidence_dict[image_path]['Confidence'] = confidence_score
        confidence_dict[image_path]['prediction_id'] = prediction_id
        confidence_dict[image_path]['imagenet_id'] = imagenet_id


for T in T_list:
    correct = 0
    wrong = 0

    for key, val in confidence_dict.items():
        confidence_score = val['Confidence']
        prediction_id = val['prediction_id']
        imagenet_id = val['imagenet_id']

        if (confidence_score > T and prediction_id == imagenet_id) or (
                confidence_score <= T and prediction_id != imagenet_id):
            correct += 1
        else:
            wrong += 1

    print("RN18 - Total: {} - Accuracy at T = {} is {}".format(correct + wrong, T, correct * 100 / (correct + wrong)))

