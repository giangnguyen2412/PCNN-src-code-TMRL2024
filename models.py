import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams

RunningParams = RunningParams()


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
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        fc_output = self.fc0_bn(torch.nn.functional.relu(self.fc0(output)))
        ds_output = self.fc1_bn(torch.nn.functional.relu(self.fc1(fc_output)))

        return ds_output, fc_output


class AdvisingNetwork(nn.Module):
    def __init__(self):
        super(AdvisingNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        conv_features = list(resnet.children())[:-1]  # delete the last fc layer

        self.resnet = nn.Sequential(*conv_features)

        resnet_hm = models.resnet18(pretrained=True)
        conv_features_hm = list(resnet_hm.children())[:-1]  # delete the last fc layer
        self.resnet_hm = nn.Sequential(*conv_features_hm)

        avg_pool_features = resnet.fc.in_features
        softmax_features = resnet.fc.out_features
        if RunningParams.XAI_method == 'No-XAI':
            fc0_in_features = avg_pool_features + softmax_features  # 512*2 + 1000
        elif RunningParams.XAI_method == 'GradCAM':
            fc0_in_features = avg_pool_features*2 + softmax_features  # 512*2 + 1000
        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc0_bn = nn.BatchNorm1d(512, eps=1e-2)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = nn.BatchNorm1d(128, eps=1e-2)
        self.fc2 = nn.Linear(128, 2)
        self.fc2_bn = nn.BatchNorm1d(2, eps=1e-2)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, heatmaps, scores):
        input_feat = self.resnet(images).squeeze()
        input_feat = input_feat / input_feat.amax(dim=1, keepdim=True)
        heatmaps = torch.cat([heatmaps, heatmaps, heatmaps], dim=1)

        explanation_feat = self.resnet_hm(heatmaps).squeeze()

        explanation_feat = explanation_feat / explanation_feat.amax(dim=1, keepdim=True)
        scores = scores / scores.amax(dim=1, keepdim=True)
        if RunningParams.XAI_method == 'No-XAI':
            concat_feat = torch.concat([input_feat, scores], dim=1)
        elif RunningParams.XAI_method == 'GradCAM':
            concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
        output = self.fc0_bn(torch.nn.functional.relu(self.fc0(concat_feat)))
        output = self.fc1_bn(torch.nn.functional.relu(self.fc1(output)))
        output = self.fc2_bn(torch.nn.functional.relu(self.fc2(output)))

        return output
