import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams

RunningParams = RunningParams()


class AdvisingNetworkv1(nn.Module):
    def __init__(self):
        super(AdvisingNetworkv1, self).__init__()
        resnet = models.resnet18(pretrained=True)
        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
        conv_features = list(resnet.children())[:-1]  # delete the last fc layer
        self.resnet = nn.Sequential(*conv_features)

        if RunningParams.heatmap_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
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
        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
        conv_features = list(resnet.children())[:-1]  # delete the last fc layer
        self.avgpool = nn.Sequential(*conv_features)

        if RunningParams.XAI_method == RunningParams.GradCAM:
            resnet_hm = models.resnet18(pretrained=True)
            if RunningParams.heatmap_frozen is True:
                for param in resnet_hm.parameters():
                    param.requires_grad = False
            conv_features_hm = list(resnet_hm.children())[:-1]  # delete the last fc layer
            self.resnet_hm = nn.Sequential(*conv_features_hm)

        elif RunningParams.XAI_method == RunningParams.NNs:
            resnet_nns = models.resnet18(pretrained=True)
            if RunningParams.nns_frozen is True:
                for param in resnet_nns.parameters():
                    param.requires_grad = False
            conv_features_nns = list(resnet_nns.children())[:-1]  # delete the last fc layer
            self.resnet_nns = nn.Sequential(*conv_features_nns)

        avg_pool_features = resnet.fc.in_features
        softmax_features = resnet.fc.out_features
        if RunningParams.XAI_method == RunningParams.NO_XAI:
            fc0_in_features = avg_pool_features + softmax_features  # 512*2 + 1000
        elif RunningParams.XAI_method == RunningParams.GradCAM:
            fc0_in_features = avg_pool_features + 7*7 + softmax_features  # 512 + heatmap_size + 1000
        elif RunningParams.XAI_method == RunningParams.NNs:
            fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features  # 512 + NN_size + 1000

        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc0_bn = nn.BatchNorm1d(512, eps=1e-2)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = nn.BatchNorm1d(128, eps=1e-2)
        self.fc2 = nn.Linear(128, 2)
        self.fc2_bn = nn.BatchNorm1d(2, eps=1e-2)

        self.input_bn = nn.BatchNorm1d(avg_pool_features, eps=1e-2)
        if RunningParams.XAI_method == RunningParams.GradCAM:
            self.exp_bn = nn.BatchNorm1d(7*7, eps=1e-2)
        elif RunningParams.XAI_method == RunningParams.NNs:
            self.exp_bn = nn.BatchNorm1d(avg_pool_features*RunningParams.k_value, eps=1e-2)
        self.scores_bn = nn.BatchNorm1d(softmax_features, eps=1e-2)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        input_feat = self.avgpool(images).squeeze()

        if RunningParams.XAI_method == RunningParams.GradCAM:
            explanation_feat = explanations.view(images.shape[0], -1)
        elif RunningParams.XAI_method == RunningParams.NNs:
            q_list = []

            for q_idx in range(explanations.shape[0]):
                exemplars = explanations[q_idx]
                # Both query and NN using the same branch
                explanation_feat = self.resnet_nns(exemplars)
                explanation_feat = explanation_feat.flatten()
                q_list.append(explanation_feat)
            explanation_feat = torch.stack(q_list)

        if RunningParams.BATCH_NORM is True:
            # input_feat = self.input_bn(torch.nn.functional.relu(input_feat))
            # if RunningParams.XAI_method != RunningParams.NO_XAI:
            #     explanation_feat = self.exp_bn(torch.nn.functional.relu(explanation_feat))
            # scores = self.scores_bn(torch.nn.functional.relu(scores))
            a = None # Remove BN because it has been just applied
        else:
            input_feat = input_feat / input_feat.amax(dim=1, keepdim=True)
            explanation_feat = explanation_feat / explanation_feat.amax(dim=1, keepdim=True)
            scores = scores / scores.amax(dim=1, keepdim=True)

        # TODO: move RunningParams.NO_XAI, ... to Explainer class
        if RunningParams.XAI_method == RunningParams.NO_XAI:
            concat_feat = torch.concat([input_feat, scores], dim=1)
        elif RunningParams.XAI_method == RunningParams.GradCAM:
            concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
        elif RunningParams.XAI_method == RunningParams.NNs:
            concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)

        output = self.fc0_bn(torch.nn.functional.relu(self.fc0(concat_feat)))
        output = self.fc1_bn(torch.nn.functional.relu(self.fc1(output)))
        output = self.fc2_bn(torch.nn.functional.relu(self.fc2(output)))

        return output, input_feat, explanation_feat


class SimpleAdvisingNetwork(nn.Module):
    def __init__(self):
        super(SimpleAdvisingNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
        conv_features = list(resnet.children())[:-4]  # delete the last fc layer
        self.avgpool = nn.Sequential(*conv_features)

        if RunningParams.XAI_method == RunningParams.GradCAM:
            resnet_hm = models.resnet18(pretrained=True)
            if RunningParams.heatmap_frozen is True:
                for param in resnet_hm.parameters():
                    param.requires_grad = False
            conv_features_hm = list(resnet_hm.children())[:-4]  # delete the last fc layer
            self.resnet_hm = nn.Sequential(*conv_features_hm)

        elif RunningParams.XAI_method == RunningParams.NNs:
            resnet_nns = models.resnet18(pretrained=True)
            if RunningParams.nns_frozen is True:
                for param in resnet_nns.parameters():
                    param.requires_grad = False
            conv_features_nns = list(resnet_nns.children())[:-4]  # delete the last fc layer
            self.resnet_nns = nn.Sequential(*conv_features_nns)

        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        avg_pool_features = 128
        softmax_features = resnet.fc.out_features
        if RunningParams.XAI_method == RunningParams.NO_XAI:
            fc0_in_features = avg_pool_features + softmax_features  # 512*2 + 1000
        elif RunningParams.XAI_method == RunningParams.GradCAM:
            fc0_in_features = avg_pool_features + 7*7 + softmax_features  # 512 + heatmap_size + 1000
        elif RunningParams.XAI_method == RunningParams.NNs:
            fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features  # 512 + NN_size + 1000

        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc0_bn = nn.BatchNorm1d(512, eps=1e-2)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = nn.BatchNorm1d(128, eps=1e-2)
        self.fc2 = nn.Linear(128, 2)
        self.fc2_bn = nn.BatchNorm1d(2, eps=1e-2)

        self.input_bn = nn.BatchNorm1d(avg_pool_features, eps=1e-2)
        if RunningParams.XAI_method == RunningParams.GradCAM:
            self.exp_bn = nn.BatchNorm1d(7*7, eps=1e-2)
        elif RunningParams.XAI_method == RunningParams.NNs:
            self.exp_bn = nn.BatchNorm1d(avg_pool_features*RunningParams.k_value, eps=1e-2)
        self.scores_bn = nn.BatchNorm1d(softmax_features, eps=1e-2)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        input_feat = self.pooling_layer(self.avgpool(images)).squeeze()

        if RunningParams.XAI_method == RunningParams.GradCAM:
            explanation_feat = explanations.view(images.shape[0], -1)
        elif RunningParams.XAI_method == RunningParams.NNs:
            q_list = []
            avg_feat = []

            for q_idx in range(explanations.shape[0]):
                exemplars = explanations[q_idx]
                # Both query and NN using the same branch
                explanation_feat = self.pooling_layer(self.resnet_nns(exemplars))
                # if RunningParams.k_value > 1:
                avg_exp_feat = torch.mean(explanation_feat, dim=0, keepdim=True).flatten()  # 1, 512

                explanation_feat = explanation_feat.flatten()

                q_list.append(explanation_feat)
                avg_feat.append(avg_exp_feat)
            explanation_feat = torch.stack(q_list)
            avg_exp_feat = torch.stack(avg_feat)

        if RunningParams.BATCH_NORM is True:
            # input_feat = self.input_bn(torch.nn.functional.relu(input_feat))
            # if RunningParams.XAI_method != RunningParams.NO_XAI:
            #     explanation_feat = self.exp_bn(torch.nn.functional.relu(explanation_feat))
            # scores = self.scores_bn(torch.nn.functional.relu(scores))
            a = None # Remove BN because it has been just applied
        else:
            input_feat = input_feat / input_feat.amax(dim=1, keepdim=True)
            explanation_feat = explanation_feat / explanation_feat.amax(dim=1, keepdim=True)
            scores = scores / scores.amax(dim=1, keepdim=True)

        # TODO: move RunningParams.NO_XAI, ... to Explainer class
        if RunningParams.XAI_method == RunningParams.NO_XAI:
            concat_feat = torch.concat([input_feat, scores], dim=1)
        elif RunningParams.XAI_method == RunningParams.GradCAM:
            concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
        elif RunningParams.XAI_method == RunningParams.NNs:
            concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)

        output = self.fc0_bn(torch.nn.functional.relu(self.fc0(concat_feat)))
        output = self.fc1_bn(torch.nn.functional.relu(self.fc1(output)))
        output = self.fc2_bn(torch.nn.functional.relu(self.fc2(output)))

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        else:
            if RunningParams.k_value > 1 and RunningParams.XAI_method == RunningParams.NNs:
                return output, input_feat, avg_exp_feat
            else:
                return output, input_feat, explanation_feat
