import torchvision
import torch.nn as nn
import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = torchvision.models.resnet18(pretrained=True).cuda()
model.fc = nn.Linear(model.fc.in_features, 196)

my_model_state_dict = torch.load(
    '/home/giang/Downloads/advising_network/PyTorch-Stanford-Cars-Baselines/model_best.pth.tar')
model.load_state_dict(my_model_state_dict['state_dict'], strict=True)
model.eval()
model.cuda()

valdir = '/home/giang/Downloads/Cars/Stanford-Cars-dataset/test'

import torchvision.transforms as transforms
import torchvision.datasets as datasets
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

running_corrects = 0
total_cnt = 0
with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):

        images, target = images.cuda(), target.cuda()

        # compute output
        output = model(images)

        p = torch.softmax(output, dim=1)

        # Compute top-1 predictions and accuracy
        score, index = torch.topk(p, 1, dim=1)

        running_corrects += torch.sum(index.squeeze() == target.cuda())
        total_cnt += images.shape[0]

        print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))

