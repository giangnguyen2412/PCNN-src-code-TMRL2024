import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams
from cross_vit import CrossTransformer, Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        self.conv_layers = nn.Sequential(*conv_features)
        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        class transformer_feat_embedder(nn.Module):
            def __init__(self, num_patches, dim):
                super().__init__()
                self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
                self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

            def forward(self, feat, k_value):
                if k_value == 1:
                    b, n, _ = feat.shape
                    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
                    x = torch.cat((cls_tokens, feat), dim=1)
                else:
                    b, k, n, _ = feat.shape
                    # cls_tokens = repeat(self.cls_token, '() () n d -> b k n d', b=b, k=k)
                    cls_tokens = repeat(self.cls_token, '() n d -> b k n d', b=b, k=k)
                    x = torch.cat((cls_tokens, feat), dim=2)

                x += self.pos_embedding[:, :(n + 1)]

                return x

        self.transformer_feat_embedder = transformer_feat_embedder(RunningParams.feat_map_size[RunningParams.conv_layer],
                                                                   RunningParams.conv_layer_size[RunningParams.conv_layer])

        self.transformer = Transformer(dim=2048, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.0)
        self.cross_transformer = CrossTransformer(sm_dim=2048, lg_dim=2048, depth=2, heads=8,
                                                  dim_head=64, dropout=0.0)

        # TODO: What is dim_head, mlp_dim? How these affect the performance?
        # Branch 1 takes softmax scores and cosine similarity
        self.branch1 = BinaryMLP(1, 512)
        # Branch 3 takes transformer features and sep_token*2
        self.branch3 = BinaryMLP(RunningParams.k_value*RunningParams.conv_layer_size[RunningParams.conv_layer]*2 +
                                 RunningParams.k_value*2, 32)

        self.dropout = nn.Dropout(p=RunningParams.dropout)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        input_spatial_feats = self.conv_layers(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()
        input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsx512x49

        if RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                explanations = explanations.squeeze()
                explanation_spatial_feats = self.conv_layers(explanations)
                explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
                explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)

                emb_cos_sim = F.cosine_similarity(input_feat, explanation_feat)
                emb_cos_sim = emb_cos_sim.unsqueeze(dim=1)
            else:  # K > 1
                explanation_spatial_feats = []
                for sample_idx in range(RunningParams.k_value):
                    data = explanations[:, 0, :]
                    explanation_spatial_feat = self.conv_layers(data)
                    explanation_spatial_feat = explanation_spatial_feat.flatten(start_dim=2)  #
                    explanation_spatial_feats.append(explanation_spatial_feat)

                # Expect to have kx2048x49
                explanation_spatial_feats = torch.stack(explanation_spatial_feats, dim=1)

        # change from 2048x49 -> 49x2048
        input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
        if RunningParams.k_value == 1:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)
        else:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 2, 3)  # 4x3x49x2048

        sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

        transformer_encoder_depth = 1

        if RunningParams.k_value == 1:
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 1)

            for _ in range(transformer_encoder_depth):
                input_spt_feats = self.transformer(input_spatial_feats)
                exp_spt_feats = self.transformer(explanation_spatial_feats)

                input_spatial_feats, explanation_spatial_feats = self.cross_transformer(input_spt_feats, exp_spt_feats)

                input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
                input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()
                # Extracting the cls token
                # TODO: We should put the cls token extraction out of transformer_encoder_depth loop if transformer_encoder_depth > 1
                # TODO: Wrong implementation
                input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
                transformer_emb = torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1)

        else:
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 3)

            transformer_embs = []
            for sample_idx in range(RunningParams.k_value):
                explanation = explanation_spatial_feats[:, sample_idx, :]
                for _ in range(transformer_encoder_depth):
                    input_spt_feats = self.transformer(input_spatial_feats)
                    explanation = self.transformer(explanation)

                    input_spatial_feats, explanation = self.cross_transformer(input_spt_feats, explanation)

                    input_emb, exp_emb = input_spatial_feats, explanation
                    input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()
                    # TODO: We should put the cls token extraction out of transformer_encoder_depth loop if transformer_encoder_depth > 1
                    # TODO: Wrong implementation
                    input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
                    transformer_emb = torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1)
                    transformer_embs.append(transformer_emb)

                transformer_emb = torch.cat(transformer_embs, dim=1)
                MERGE_CONF = False
                if MERGE_CONF:
                    # transformer_emb = torch.cat([transformer_emb, sep_token, scores], dim=1)
                    transformer_emb = torch.cat([transformer_emb, sep_token, scores.amax(dim=1, keepdim=True)], dim=1)

        # output1 = self.branch1(torch.cat([emb_cos_sim], dim=1))
        output3 = self.branch3(transformer_emb)
        output = output3

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        elif RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                return output, input_feat, explanation_feat, emb_cos_sim
            else:
                return output, input_feat, None, None