import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
from torchvision import models
from params import RunningParams
from cross_vit import CrossTransformer, Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision

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


# class CUB200MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 200),  # 2 for binary classification
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)

class CUB200MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
        )

    def forward(self, x):
        return self.net(x)



class Transformer_AdvisingNetwork(nn.Module):
    def __init__(self):
        super(Transformer_AdvisingNetwork, self).__init__()

        if RunningParams.CUB_TRAINING is True:
            from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
            resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
            my_model_state_dict = torch.load(
                'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
            resnet.load_state_dict(my_model_state_dict, strict=True)
        elif RunningParams.DOGS_TRAINING is True:
            resnet = torchvision.models.resnet50(pretrained=True).cuda()

            if RunningParams.SIMCLR_MODEL is True:
                from modelvshuman.models.pytorch.simclr import simclr_resnet50x1
                resnet = simclr_resnet50x1(pretrained=True, use_data_parallel=False)

        if RunningParams.CUB_200WAY is True:
            from collections import OrderedDict

            resnet = torchvision.models.resnet50(pretrained=True).cuda()
            my_model_state_dict = torch.load("/home/giang/Downloads/Cub-ResNet-iNat/resnet50_inat_pretrained_0.841.pth")
            my_model_state_dict = OrderedDict(
                {name.replace("layers.", ""): value for name, value in my_model_state_dict.items()}
            )
            resnet.load_state_dict(my_model_state_dict, strict=False)

        if RunningParams.query_frozen is True:
            for param in resnet.parameters():
                param.requires_grad = False

        conv_features = list(resnet.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
        self.conv_layers = nn.Sequential(*conv_features)
        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        class TransformerFeatureEmbedder(nn.Module):
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

        self.TransformerFeatureEmbedder = TransformerFeatureEmbedder(RunningParams.feat_map_size[RunningParams.conv_layer],
                                                                   RunningParams.conv_layer_size[RunningParams.conv_layer])

        if RunningParams.CUB_200WAY is True:
            self.transformer = Transformer(dim=2048, depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=0.0)
            self.cross_transformer = CrossTransformer(sm_dim=2048, lg_dim=2048, depth=1, heads=8,
                                                      dim_head=64, dropout=0.0)
        else:
            self.transformer = Transformer(dim=2048, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.0)
            self.cross_transformer = CrossTransformer(sm_dim=2048, lg_dim=2048, depth=2, heads=8,
                                                      dim_head=64, dropout=0.0)

        # Branch 3 takes transformer features and sep_token*2
        if RunningParams.CUB_200WAY is True:
            # self.branch3 = CUB200MLP(RunningParams.k_value*RunningParams.conv_layer_size[RunningParams.conv_layer]*2 +
            #                      RunningParams.k_value*2, 512)
            self.branch3 = CUB200MLP(
                (RunningParams.k_value*2) * RunningParams.conv_layer_size[RunningParams.conv_layer] + RunningParams.k_value*2, None)
        else:
            self.branch3 = BinaryMLP(RunningParams.k_value*RunningParams.conv_layer_size[RunningParams.conv_layer]*2 +
                                     RunningParams.k_value*2, 32)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        input_spatial_feats = self.conv_layers(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()
        input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsxcx49

        if RunningParams.k_value == 1:
            explanations = explanations.squeeze()
            explanation_spatial_feats = self.conv_layers(explanations)
            explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
            explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)
        else:  # K > 1
            explanation_spatial_feats = []
            explanation_feats = []
            # Process sample by sample
            for input_idx in range(images.shape[0]):
                data = explanations[input_idx, :, :, :, :]
                explanation_spatial_feat = self.conv_layers(data)
                explanation_feat = self.pooling_layer(explanation_spatial_feat).squeeze()  # kx2048
                # TODO: I must check these two when training the advising networks
                explanation_spatial_feat = explanation_spatial_feat.flatten(start_dim=2)  # kx2048x49
                explanation_spatial_feats.append(explanation_spatial_feat)
                ## TODO:
                explanation_feats.append(explanation_feat.flatten())

            # Expect to have kx2048x49
            explanation_spatial_feats = torch.stack(explanation_spatial_feats, dim=0)
            explanation_feat = torch.stack(explanation_feats, dim=0)

        # if RunningParams.CUB_200WAY is True:
        #     features = torch.cat((input_feat, explanation_feat), dim=1)
        #     output = self.branch3(features)
        #     return output, input_feat

        # change from 2048x49 -> 49x2048
        input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
        if RunningParams.k_value == 1:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)
        else:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 2, 3)  # 4x3x49x2048

        sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

        if RunningParams.k_value == 1:
            input_spatial_feats = self.TransformerFeatureEmbedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.TransformerFeatureEmbedder(explanation_spatial_feats, 1)

            input_spatial_feats = self.transformer(input_spatial_feats)
            explanation_spatial_feats = self.transformer(explanation_spatial_feats)

            input_spatial_feats, explanation_spatial_feats = self.cross_transformer(input_spatial_feats, explanation_spatial_feats)

            input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
            input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

            # Extracting the cls token
            input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
            transformer_emb = torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1)

        else:
            input_spatial_feats = self.TransformerFeatureEmbedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.TransformerFeatureEmbedder(explanation_spatial_feats, 3)

            input_spatial_feats = self.transformer(input_spatial_feats)
            transformer_embs = []
            # We need to loop over prototypes because we need CA b/w query vs. each NN
            for expl_idx in range(RunningParams.k_value):
                explanation = explanation_spatial_feats[:, expl_idx, :, :]
                explanation = self.transformer(explanation)
                # TODO: uncomment the two lines below to enable CA
                input, explanation = self.cross_transformer(input_spatial_feats, explanation)
                input_emb, exp_emb = input, explanation
                # TODO: uncomment the 1 line below to enable CA
                # input_emb, exp_emb = input_spatial_feats, explanation
                input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

                input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
                transformer_emb = torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1)
                transformer_embs.append(transformer_emb)

            transformer_emb = torch.cat(transformer_embs, dim=1)

        output = self.branch3(transformer_emb)

        return output, input_feat