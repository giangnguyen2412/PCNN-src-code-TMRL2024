import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

class CustomViT(nn.Module):
    def __init__(self, base_model):
        super(CustomViT, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # Get the features from the base ViT model
        x = self.base_model.forward_features(x)
        # Extract the CLS token (first token)
        cls_token = x[:, 0]
        # Pass the features through the classifier
        output = self.base_model.head(cls_token)
        return output, cls_token

import os
from params import RunningParams

RunningParams = RunningParams()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Parameters
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CUB-200 validation dataset
val_dataset = datasets.ImageFolder(root=f'{RunningParams.parent_dir}/{RunningParams.test_path}', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

# Initialize the base model and load the trained weights
base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=200)
model_path = "./vit_base_patch16_224_cub_200way_82_40.pth"
state_dict = torch.load(model_path, map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
base_model.load_state_dict(new_state_dict)

# Wrap the base model in the custom model
model = CustomViT(base_model).to(device)
model.eval()

# Count the parameters
total_params = sum(p.numel() for p in model.parameters())

print(f'Total trainable parameters: {total_params}')

# Run validation and extract CLS tokens
cls_tokens = []
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, tokens = model(inputs)
        cls_tokens.append(tokens.cpu())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# CLS tokens are stored in cls_tokens
