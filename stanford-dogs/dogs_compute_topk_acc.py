import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
from PIL import Image
from params import RunningParams

RunningParams = RunningParams('DOGS')

import torchvision
from dataloader import StanfordDogsDataset  # Ensure you have this dataloader

# Choose the ResNet model: 'resnet18', 'resnet34', or 'resnet50'
model_type = f'resnet{RunningParams.resnet}'


if RunningParams.resnet == 50:
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model.fc = nn.Linear(2048, 120)
elif RunningParams.resnet == 34:
    model = torchvision.models.resnet34(pretrained=True).cuda()
    model.fc = nn.Linear(512, 120)
elif RunningParams.resnet == 18:
    model = torchvision.models.resnet18(pretrained=True).cuda()
    model.fc = nn.Linear(512, 120)

my_model_state_dict = torch.load(
    f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth',
    map_location='cuda'
)
new_state_dict = {k.replace("model.", ""): v for k, v in my_model_state_dict.items()}

model.load_state_dict(new_state_dict, strict=True)
model.eval().cuda()

# Define the custom dataset class
class StanfordDogsDataset(Dataset):
    def __init__(self, file_list, root_dir, breed_to_label, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.breed_to_label = breed_to_label
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx][0][0])
        image = Image.open(img_name)

        # Extract label from the directory name
        breed = img_name.split('/')[-2]
        label = self.breed_to_label[breed]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the path to your dataset and .mat files
dataset_root = '/home/giang/Downloads/Stanford_Dogs_dataset'  # Update this path
train_mat = scipy.io.loadmat(os.path.join(dataset_root, 'train_list.mat'))
test_mat = scipy.io.loadmat(os.path.join(dataset_root, 'test_list.mat'))

# Extract file names and labels
train_files = train_mat['file_list']
test_files = test_mat['file_list']

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # Convert images to RGB
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create a mapping of breed names to integer labels
breed_names = sorted(os.listdir(os.path.join(dataset_root, 'Images')))
breed_to_label = {breed: i for i, breed in enumerate(breed_names)}

# Create datasets
train_dataset = StanfordDogsDataset(train_files, os.path.join(dataset_root, 'Images'), breed_to_label, transform=preprocess)
test_dataset = StanfordDogsDataset(test_files, os.path.join(dataset_root, 'Images'), breed_to_label, transform=preprocess)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

import torch

def validate_topk(model, loader, topk=(1,)):
    model.eval()  # Set the model to evaluation mode
    maxk = max(topk)
    num_classes = 120  # Number of classes in Stanford Dogs dataset

    topk_corrects = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # Calculate top-k class indices for each image
            _, topk_indices = outputs.topk(maxk, 1, True, True)

            correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))

            for k in topk:
                topk_corrects[k] += correct[:, :k].reshape(-1).float().sum(0, keepdim=True).item()

            total += labels.size(0)

    accuracies = {k: 100.0 * topk_corrects[k] / total for k in topk}
    return accuracies

# Usage
topk = tuple(range(1, 11))  # Top-1 to Top-10
topk_accuracies = validate_topk(model, test_loader, topk=topk)

print('-' * 10)
for k in topk:
    print(f'Top-{k} Accuracy: {topk_accuracies[k]:.2f}%')