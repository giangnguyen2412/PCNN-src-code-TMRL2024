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

RunningParams = RunningParams()

from dataloader import StanfordDogsDataset  # Ensure you have this dataloader

# Choose the ResNet model: 'resnet18', 'resnet34', or 'resnet50'
model_type = f'resnet{RunningParams.resnet}'

class FineTuneResNet(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()
        if model_type == 'resnet18':
            self.model = resnet18(pretrained=True)
        elif model_type == 'resnet34':
            self.model = resnet34(pretrained=True)
        elif model_type == 'resnet50':
            self.model = resnet50(pretrained=True)
        else:
            raise ValueError("Invalid model type. Choose 'resnet18', 'resnet34', or 'resnet50'.")

        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

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

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FineTuneResNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Validation function
def validate(model, loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_accuracy = validate(model, test_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Finished Training')
torch.save(model.state_dict(), f"./{model_type}_stanford_dogs.pth")