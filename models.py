import torch
import torch.nn as nn
from torchvision import models


class MyCustomResnet18(nn.Module):
    def __init__(self, pretrained=True, fine_tune=False):
        super().__init__()

        resnet18 = models.resnet18(pretrained=pretrained)
        if fine_tune is True:
            for param in resnet18.parameters():
                param.requires_grad = False

        self.features = nn.ModuleList(resnet18.children())[:-1]
        self.features = nn.Sequential(*self.features)
        in_features = resnet18.fc.in_features
        self.fc0 = nn.Linear(in_features, 1000)
        self.fc0_bn = nn.BatchNorm1d(1000, eps=1e-2)
        self.fc1 = nn.Linear(1000, 2)
        self.fc1_bn = nn.BatchNorm1d(2, eps=1e-2)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        fc_output = self.fc0_bn(torch.nn.functional.relu(self.fc0(output)))
        ds_output = self.fc1_bn(torch.nn.functional.relu(self.fc1(fc_output)))

        return ds_output, fc_output
