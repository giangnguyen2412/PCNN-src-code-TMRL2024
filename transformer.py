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


class Transformer_AdvisingNetwork(nn.Module):
    def __init__(self):
        super(Transformer_AdvisingNetwork, self).__init__()

        # resnet = torchvision.models.resnet50(pretrained=True).cuda()
        # resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
        # my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
        # resnet.load_state_dict(my_model_state_dict, strict=True)
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
                    # TODO: Need to modify to assign cls as the avg value of feat
                    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
                    x = torch.cat((cls_tokens, feat), dim=1)
                else:
                    b, k, n, _ = feat.shape
                    cls_tokens = repeat(self.cls_token, '() n d -> b k n d', b=b, k=k)
                    # TODO: I revert back to the original to train again.
                    # TODO: Later I will inspect the model to see if the attention makes sense. If it does not, then I will use the mean
                    # cls_tokens = torch.mean(feat, dim=2, keepdim=True)
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
        # self.branch1 = BinaryMLP(200, 2)
        # Branch 3 takes transformer features and sep_token*2
        self.branch3 = BinaryMLP(RunningParams.k_value*RunningParams.conv_layer_size[RunningParams.conv_layer]*2 +
                                 RunningParams.k_value*2, 32)

        # self.output_layer = BinaryMLP(2 * 2 + 2, 2)

        self.dropout = nn.Dropout(p=RunningParams.dropout)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, images, explanations, scores):
        # Process the input images
        input_spatial_feats = self.conv_layers(images)
        input_feat = self.pooling_layer(input_spatial_feats).squeeze()
        input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsxcx49

        # Process the nearest neighbors
        if RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                explanations = explanations.squeeze()
                explanation_spatial_feats = self.conv_layers(explanations)
                explanation_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
                explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)
            else:  # K > 1
                explanation_spatial_feats = []
                for sample_idx in range(RunningParams.k_value):
                    data = explanations[:, sample_idx, :]  # TODO: I traced to be at least correct here
                    explanation_spatial_feat = self.conv_layers(data)
                    explanation_spatial_feat = explanation_spatial_feat.flatten(start_dim=2)  #
                    explanation_spatial_feats.append(explanation_spatial_feat)

                # bsxKx2048x49
                explanation_spatial_feats = torch.stack(explanation_spatial_feats, dim=1)

        # change from 2048x49 -> 49x2048
        # 49 tokens for an image
        input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
        if RunningParams.k_value == 1:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)
        else:
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 2, 3)  # 4x3x49x2048

        sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

        transformer_encoder_depth = 2

        if RunningParams.k_value == 1:
            # Add the cls token and positional embedding --> 50x2048
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 1)

            # TODO: as the output of the first layer does not propagate to the second layer, then transformer_encoder_depth should be 1
            for _ in range(transformer_encoder_depth):
                # Self-attention --> 50x2048; both cls and image tokens are transformed.
                input_spt_feats = self.transformer(input_spatial_feats)
                exp_spt_feats = self.transformer(explanation_spatial_feats)

                # Cross-attention --> 50x2048; only the cls tokens are transformed. Image tokens are kept the same.
                input_spatial_feats, explanation_spatial_feats = self.cross_transformer(input_spt_feats, exp_spt_feats)

            input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
            input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

            # Extracting the cls token --> 1x2048
            input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))
            transformer_emb = torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1)

        else:
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)  # bsx50x2048
            # Clone the input tensor K times along the second dimension
            input_spatial_feats = input_spatial_feats.unsqueeze(1).repeat(1, RunningParams.k_value, 1, 1).\
                view(input_spatial_feats.shape[0], RunningParams.k_value, 50, 2048)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 3)  # bsxKx50x2048

            i2e_attns = []
            e2i_attns = []
            for depth_idx in range(transformer_encoder_depth):
                input_list = []
                explanation_list = []
                for prototype_idx in range(RunningParams.k_value):
                    input = input_spatial_feats[:, prototype_idx, :]
                    explanation = explanation_spatial_feats[:, prototype_idx, :]

                    input = self.transformer(input)
                    explanation = self.transformer(explanation)

                    # TODO: Thử remove self-attention
                    # TODO: với VIT hiện tại ko có model về patch (local) mà chỉ có global info
                    # input = input_spatial_feats
                    # explanation = explanation

                    # Cross-attention --> bsx50x2048; only the cls tokens are transformed. Image tokens are kept the same.
                    inp, exp, i2e_attn, e2i_attn = self.cross_transformer(input, explanation)

                    # Extract attention from the last layer (i.e. closest to classification head)
                    if depth_idx == transformer_encoder_depth - 1:
                        i2e_attns.append(i2e_attn)
                        e2i_attns.append(e2i_attn)

                    input_list.append(inp)
                    explanation_list.append(exp)

                input_spatial_feats = torch.stack(input_list, dim=1)
                explanation_spatial_feats = torch.stack(explanation_list, dim=1)

            input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
            input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

            input_cls = input_emb[:, :, 0]
            exp_cls = exp_emb[:, :, 0]

            transformer_emb = torch.cat([sep_token, input_cls[:, 0, :], sep_token, exp_cls[:, 0, :]], dim=1)
            for prototype_idx in range(1, RunningParams.k_value):
                transformer_emb = torch.cat([transformer_emb, sep_token, input_cls[:, prototype_idx], sep_token, exp_cls[:, prototype_idx]], dim=1)

            i2e_attns = torch.cat(i2e_attns, dim=2)
            e2i_attns = torch.cat(e2i_attns, dim=2)

        output3 = self.branch3(transformer_emb)
        # output1 = self.branch1(torch.cat([scores], dim=1))
        #
        # output1_p = torch.nn.functional.softmax(output1, dim=1)
        # output3_p = torch.nn.functional.softmax(output3, dim=1)
        #
        # output = torch.cat([sep_token, output1_p, sep_token, output3_p], dim=1)
        # output = self.output_layer(output)
        output = output3

        if RunningParams.XAI_method == RunningParams.NO_XAI:
            return output, None, None
        elif RunningParams.XAI_method == RunningParams.NNs:
            if RunningParams.k_value == 1:
                return output, input_feat, None, None
            else:
                return output, input_feat, i2e_attns, e2i_attns