import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm  # PyTorch Image Models
import os
from params import RunningParams

RunningParams = RunningParams()

# Parameters
batch_size = 256
learning_rate = 3e-4
num_epochs = 20


def train():
    # Specify the GPUs to use
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

    # Data augmentation and normalization for training
    # Just normalization for validation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CUB-200 dataset
    train_dataset = datasets.ImageFolder(root=f'{RunningParams.parent_dir}/{RunningParams.train_path}', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16,
                                               pin_memory=True)

    val_dataset = datasets.ImageFolder(root=f'{RunningParams.parent_dir}/{RunningParams.test_path}', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=False)

    # Initialize the model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=200)
    model = nn.DataParallel(model).cuda()

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation accuracy
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')
    torch.save(model.state_dict(), "./vit_base_patch16_224_cub_200way_new.pth")


def main():
    train()


if __name__ == "__main__":
    print("batch_size: ", batch_size)
    main()

