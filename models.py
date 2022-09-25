import torch.nn.functional as F
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

        if RunningParams.SIMCLR_MODEL is True:
            from modelvshuman.models.pytorch.simclr import simclr_resnet50x1
            resnet = simclr_resnet50x1(pretrained=True, use_data_parallel=False)

        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
        conv_features = list(resnet.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
        self.avgpool = nn.Sequential(*conv_features)

        if RunningParams.XAI_method == RunningParams.GradCAM:
            resnet_hm = models.resnet18(pretrained=True)
            if RunningParams.heatmap_frozen is True:
                for param in resnet_hm.parameters():
                    param.requires_grad = False
            conv_features_hm = list(resnet_hm.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
            self.resnet_hm = nn.Sequential(*conv_features_hm)

        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        avg_pool_features = RunningParams.conv_layer_size[RunningParams.conv_layer]
        softmax_features = resnet.fc.out_features
        if RunningParams.USING_SOFTMAX:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                fc0_in_features = avg_pool_features + softmax_features  # avgavg_pool_size_pool + 1000
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                fc0_in_features = avg_pool_features + 7*7 + softmax_features  # avg_pool_size + heatmap_size + 1000
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + softmax_features \
                                      + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2 + 1
                else:
                    fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features + 1
        else:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                fc0_in_features = avg_pool_features
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                fc0_in_features = avg_pool_features + 7 * 7
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    # fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + \
                    fc0_in_features = 1
                else:
                    fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + 1

        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc0_bn = nn.BatchNorm1d(512, eps=1e-2)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = nn.BatchNorm1d(128, eps=1e-2)
        self.fc2 = nn.Linear(128, 2)
        self.fc2_bn = nn.BatchNorm1d(2, eps=1e-2)

        self.dropout = nn.Dropout(p=RunningParams.dropout)

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
        input_spatial_feats = self.avgpool(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()
        input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsx512x48

        if RunningParams.XAI_method == RunningParams.GradCAM:
            explanation_feat = explanations.view(images.shape[0], -1)
        elif RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                explanations = explanations.squeeze()
                explanation_spatial_feats = self.avgpool(explanations)
                explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
                explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)

                if RunningParams.CrossCorrelation is True:
                    # b: batch_size, c: channel nums, m and n is the size of feature maps (e.g. 7x7=49)
                    attention_vectors = torch.einsum('bcm,bcn->bmn', explanation_spatial_feats, input_spatial_feats)
                    attention_vectors = attention_vectors.flatten(start_dim=1)
            else:  # K > 1
                explanation_feats = []
                attention_vectors = []
                for sample_idx in range(explanations.shape[0]):
                    data = explanations[sample_idx]
                    explanation_spatial_feats = self.avgpool(data)

                    explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()  # 3x512
                    explanation_feat = explanation_feat.flatten()  # 1536
                    explanation_feats.append(explanation_feat)

                    explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)  # 3x512x49

                    if RunningParams.CrossCorrelation is True:
                        input_spatial_feat = input_spatial_feats[sample_idx]
                        # k: k neighbors, c: channel nums, m and n is the size of feature maps (e.g. 7x7=49)
                        attention_vector = torch.einsum('kcm,cn->kmn', explanation_spatial_feats, input_spatial_feat)
                        attention_vector = attention_vector.flatten()  # 2401*K
                        attention_vectors.append(attention_vector)

                explanation_feat = torch.stack(explanation_feats)
                if RunningParams.CrossCorrelation is True:
                    attention_vectors = torch.stack(attention_vectors)

        emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
        emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
        # import pdb
        # pdb.set_trace()

        if RunningParams.USING_SOFTMAX is True:
            # TODO: move RunningParams.NO_XAI, ... to Explainer class
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                concat_feat = torch.concat([input_feat, scores], dim=1)
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    concat_feat = torch.concat([emb_cos_sim, input_feat, explanation_feat, attention_vectors, scores], dim=1)
                else:
                    concat_feat = torch.concat([emb_cos_sim, input_feat, explanation_feat, scores], dim=1)
        else:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                concat_feat = input_feat
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                concat_feat = torch.concat([input_feat, explanation_feat], dim=1)
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    # concat_feat = torch.concat([emb_cos_sim, input_feat, explanation_feat, attention_vectors], dim=1)
                    concat_feat = torch.concat([emb_cos_sim], dim=1)
                else:
                    concat_feat = torch.concat([emb_cos_sim, input_feat, explanation_feat], dim=1)

        output = self.dropout(self.fc0_bn(torch.nn.functional.relu(self.fc0(concat_feat))))
        output = self.dropout(self.fc1_bn(torch.nn.functional.relu(self.fc1(output))))
        output = self.dropout(self.fc2_bn(torch.nn.functional.relu(self.fc2(output))))

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        else:
            return output, input_feat, explanation_feat


