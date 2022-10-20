import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams
from cross_vit import CrossTransformer

RunningParams = RunningParams()


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
        if RunningParams.USING_SOFTMAX is True:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                fc0_in_features = avg_pool_features + softmax_features  # avgavg_pool_size_pool + 1000
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                fc0_in_features = avg_pool_features + 7*7 + softmax_features  # avg_pool_size + heatmap_size + 1000
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + softmax_features \
                                      + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2 + 1
                else:
                    fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features
        else:
            if RunningParams.COSINE_ONLY is True:
                # fc0_in_features = RunningParams.k_value
                fc0_in_features = 1
            else:
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    fc0_in_features = avg_pool_features
                elif RunningParams.XAI_method == RunningParams.GradCAM:
                    fc0_in_features = avg_pool_features + 7 * 7
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.CrossCorrelation is True:
                        fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + \
                                          + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2
                    else:
                        fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value

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

                if RunningParams.COSINE_ONLY is True:
                    emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
                    emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
            else:  # K > 1
                explanation_feats = []
                attention_vectors = []
                emb_cos_sims = []
                for sample_idx in range(explanations.shape[0]):
                    data = explanations[sample_idx]
                    explanation_spatial_feats = self.avgpool(data)
                    explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()  # 3x512
                    if RunningParams.COSINE_ONLY is True:
                        input_tensor = input_feat[sample_idx]
                        input_tensor = input_tensor.repeat([RunningParams.k_value, 1])  # Repeat input tensors to compute cosine sim
                        emb_cos_sim = F.cosine_similarity(input_tensor, explanation_feat, dim=1)
                        # TODO:
                        # 1.debug here to see if the cosine similarity scores make sense?
                        # 2. giu k=1, thay NN la NN thu hai xem cosine sim co thap dot ngot hay k
                        # emb_cos_sims.append(emb_cos_sim)
                        emb_cos_sims.append(emb_cos_sim.mean().unsqueeze(0))

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
                if RunningParams.COSINE_ONLY is True and RunningParams.k_value > 1:
                    emb_cos_sim = torch.stack(emb_cos_sims)

        # emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
        # emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
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
                    concat_feat = torch.concat([input_feat, explanation_feat, attention_vectors, scores, emb_cos_sim], dim=1)
                else:
                    concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
        else:
            if RunningParams.COSINE_ONLY is True:
                concat_feat = torch.concat([emb_cos_sim], dim=1)
            else:
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    concat_feat = input_feat
                elif RunningParams.XAI_method == RunningParams.GradCAM:
                    concat_feat = torch.concat([input_feat, explanation_feat], dim=1)
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.CrossCorrelation is True:
                        concat_feat = torch.concat([input_feat, explanation_feat, attention_vectors], dim=1)
                        # concat_feat = torch.concat([emb_cos_sim], dim=1)
                    else:
                        concat_feat = torch.concat([input_feat, explanation_feat], dim=1)

        output = self.dropout(self.fc0_bn(torch.nn.functional.relu(self.fc0(concat_feat))))
        output = self.dropout(self.fc1_bn(torch.nn.functional.relu(self.fc1(output))))
        output = self.dropout(self.fc2_bn(torch.nn.functional.relu(self.fc2(output))))

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        else:
            return output, input_feat, explanation_feat, emb_cos_sim


class TransformerAdvisingNetwork(nn.Module):
    def __init__(self):
        super(TransformerAdvisingNetwork, self).__init__()
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
        if RunningParams.USING_SOFTMAX is True:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                fc0_in_features = avg_pool_features + softmax_features  # avgavg_pool_size_pool + 1000
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                fc0_in_features = avg_pool_features + 7*7 + softmax_features  # avg_pool_size + heatmap_size + 1000
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + softmax_features \
                                      + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2
                else:
                    fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features
        else:
            if RunningParams.COSINE_ONLY is True:
                # fc0_in_features = RunningParams.k_value
                fc0_in_features = 1
            else:
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    fc0_in_features = avg_pool_features
                elif RunningParams.XAI_method == RunningParams.GradCAM:
                    fc0_in_features = avg_pool_features + 7 * 7
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.CrossCorrelation is True:
                        fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + \
                                          + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2
                    else:
                        fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value

        self.cross_transformer = CrossTransformer(sm_dim=2048, lg_dim=2048, depth=1, heads=8, dim_head=64, dropout=0.0).cuda()

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

                if RunningParams.COSINE_ONLY is True:
                    emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
                    emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
            else:  # K > 1
                explanation_feats = []
                attention_vectors = []
                emb_cos_sims = []
                for sample_idx in range(explanations.shape[0]):
                    data = explanations[sample_idx]
                    explanation_spatial_feats = self.avgpool(data)
                    explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()  # 3x512
                    if RunningParams.COSINE_ONLY is True:
                        input_tensor = input_feat[sample_idx]
                        input_tensor = input_tensor.repeat([RunningParams.k_value, 1])  # Repeat input tensors to compute cosine sim
                        emb_cos_sim = F.cosine_similarity(input_tensor, explanation_feat, dim=1)
                        # TODO:
                        # 1.debug here to see if the cosine similarity scores make sense?
                        # 2. giu k=1, thay NN la NN thu hai xem cosine sim co thap dot ngot hay k
                        # emb_cos_sims.append(emb_cos_sim)
                        emb_cos_sims.append(emb_cos_sim.mean().unsqueeze(0))

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
                if RunningParams.COSINE_ONLY is True and RunningParams.k_value > 1:
                    emb_cos_sim = torch.stack(emb_cos_sims)

        if RunningParams.USING_SOFTMAX is True:
            # TODO: move RunningParams.NO_XAI, ... to Explainer class
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                concat_feat = torch.concat([input_feat, scores], dim=1)
            elif RunningParams.XAI_method == RunningParams.GradCAM:
                concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    concat_feat = torch.concat([input_feat, explanation_feat, attention_vectors, scores], dim=1)
                else:
                    concat_feat = torch.concat([input_feat, explanation_feat, scores], dim=1)
        else:
            if RunningParams.COSINE_ONLY is True:
                concat_feat = torch.concat([emb_cos_sim], dim=1)
            else:
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    concat_feat = input_feat
                elif RunningParams.XAI_method == RunningParams.GradCAM:
                    concat_feat = torch.concat([input_feat, explanation_feat], dim=1)
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.CrossCorrelation is True:
                        concat_feat = torch.concat([input_feat, explanation_feat, attention_vectors], dim=1)
                        # concat_feat = torch.concat([emb_cos_sim], dim=1)
                    else:
                        concat_feat = torch.concat([input_feat, explanation_feat], dim=1)

        output = self.transformer_encoder(concat_feat)
        output = self.dropout(self.fc0_bn(torch.nn.functional.relu(self.fc0(output))))
        output = self.dropout(self.fc1_bn(torch.nn.functional.relu(self.fc1(output))))
        output = self.dropout(self.fc2_bn(torch.nn.functional.relu(self.fc2(output))))

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        else:
            return output, input_feat, explanation_feat, emb_cos_sim