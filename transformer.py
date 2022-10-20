import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams
from cross_vit import CrossTransformer, Transformer

RunningParams = RunningParams()


class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 for binary classification
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer_AdvisingNetwork(nn.Module):
    def __init__(self):
        super(Transformer_AdvisingNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)

        if RunningParams.SIMCLR_MODEL is True:
            from modelvshuman.models.pytorch.simclr import simclr_resnet50x1
            resnet = simclr_resnet50x1(pretrained=True, use_data_parallel=False)

        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False
        conv_features = list(resnet.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
        self.avgpool = nn.Sequential(*conv_features)
        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        avg_pool_features = RunningParams.conv_layer_size[RunningParams.conv_layer]
        softmax_features = resnet.fc.out_features

        if RunningParams.USING_SOFTMAX is True:
            if RunningParams.XAI_method == RunningParams.NO_XAI:
                fc0_in_features = avg_pool_features + softmax_features  # avgavg_pool_size_pool + 1000
            elif RunningParams.XAI_method == RunningParams.NNs:
                if RunningParams.CrossCorrelation is True:
                    fc0_in_features = avg_pool_features + avg_pool_features * RunningParams.k_value + softmax_features \
                                      + RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2
                else:
                    fc0_in_features = avg_pool_features + avg_pool_features*RunningParams.k_value + softmax_features

        self.transformer = Transformer(dim=2048, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.0)
        self.cross_transformer = CrossTransformer(sm_dim=2048, lg_dim=2048, depth=2, heads=8,
                                                  dim_head=64, dropout=0.0)

        # TODO: What is dim_head, mlp_dim? How these affect the performance?
        # Branch 1 takes softmax scores and cosine similarity
        self.branch1 = BinaryMLP(1, 512)
        # Branch 2 takes CC map and sep_token
        self.branch2 = BinaryMLP(RunningParams.k_value*RunningParams.feat_map_size[RunningParams.conv_layer]**2, 1024)
        # Branch 3 takes transformer features and sep_token
        self.branch3 = BinaryMLP(RunningParams.k_value*RunningParams.conv_layer_size[RunningParams.conv_layer]*2 + 1, 32)

        self.dropout = nn.Dropout(p=RunningParams.dropout)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        input_spatial_feats = self.avgpool(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()
        input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsx512x49

        if RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                explanations = explanations.squeeze()
                explanation_spatial_feats = self.avgpool(explanations)
                explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
                explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)

                if RunningParams.CrossCorrelation is True:
                    # b: batch_size, c: channel nums, m and n is the size of feature maps (e.g. 7x7=49)
                    attention_vectors = torch.einsum('bcm,bcn->bmn', explanation_spatial_feats, input_spatial_feats)
                    attention_vectors = attention_vectors.flatten(start_dim=1)

                emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
                emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
            else:  # K > 1
                pass

        # change from 2048x49 -> 49x2048
        input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
        explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)

        transformer_encoder_depth = 1

        for _ in range(transformer_encoder_depth):
            input_spt_feats = self.transformer(input_spatial_feats)
            exp_spt_feats = self.transformer(explanation_spatial_feats)

            input_spatial_feats, explanation_spatial_feats = self.cross_transformer(input_spt_feats, exp_spt_feats)

        input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
        input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()
        # Extracting the cls token
        input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
        sep_token = torch.zeros([input_spt_feats.shape[0], 1], requires_grad=False).cuda()
        transformer_emb = torch.cat([input_cls, sep_token, exp_cls], dim=1)

        output1 = self.branch1(torch.cat([emb_cos_sim], dim=1))
        output2 = self.branch2(attention_vectors)
        output3 = self.branch3(transformer_emb)
        output = output3

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        else:
            return output, input_feat, explanation_feat, emb_cos_sim