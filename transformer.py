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
    def __init__(self, input_dim, hidden_dim, dropout=0.0):  # Set default dropout to 0.2
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Make sure to use `hidden_dim` instead of fixed number
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


if RunningParams.CUB_TRAINING is True:
    class Transformer_AdvisingNetwork(nn.Module):
        def __init__(self):
            print("Using training network for Birds (CUB-200)")
            super(Transformer_AdvisingNetwork, self).__init__()

            from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck
            resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')

            if RunningParams.resnet == 50 and RunningParams.RN50_INAT is False:
                resnet = models.resnet50(pretrained=True)
                resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
                my_model_state_dict = torch.load(
                    f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet50_cub_200way_top1acc_63.pth')
            elif RunningParams.resnet == 34:
                resnet = models.resnet34(pretrained=True)
                resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
                my_model_state_dict = torch.load(
                    f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet34_cub_200way_top1acc_62_81.pth')
            elif RunningParams.resnet == 18:
                resnet = models.resnet18(pretrained=True)
                resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
                my_model_state_dict = torch.load(
                    f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet18_cub_200way_top1acc_60_22.pth')

            resnet.load_state_dict(my_model_state_dict, strict=True)
            if RunningParams.resnet == 34 or RunningParams.resnet == 18 or (
                    RunningParams.resnet == 50 and RunningParams.RN50_INAT is False):
                resnet.fc = resnet.fc[0]

            conv_features = list(resnet.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
            self.conv_layers = nn.Sequential(*conv_features)

            self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            class transformer_feat_embedder(nn.Module):
                def __init__(self, num_patches, dim):
                    super().__init__()
                    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
                    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

                def forward(self, feat, k_value):
                    b, n, _ = feat.shape
                    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
                    x = torch.cat((cls_tokens, feat), dim=1)

                    x += self.pos_embedding[:, :(n + 1)]
                    return x

            self.transformer_feat_embedder = transformer_feat_embedder(RunningParams.feat_map_size[RunningParams.conv_layer],
                                                                       RunningParams.conv_layer_size[RunningParams.conv_layer])

            transformer_depth = RunningParams.N
            cross_transformer_depth = RunningParams.M
            feat_dim = RunningParams.conv_layer_size[RunningParams.conv_layer]
            self.transformer = Transformer(dim=feat_dim, depth=transformer_depth, heads=8, dim_head=64, mlp_dim=512, dropout=0.0)
            self.cross_transformer = CrossTransformer(sm_dim=feat_dim, lg_dim=feat_dim, depth=cross_transformer_depth, heads=8,
                                                      dim_head=64, dropout=0.0)

            print('Using transformer with depth {} and 8 heads'.format(transformer_depth))
            print('Using cross_transformer with depth {} and 8 heads'.format(cross_transformer_depth))

            self.branch3 = BinaryMLP(
                2 * RunningParams.conv_layer_size[RunningParams.conv_layer] + 2, 32)

            self.agg_branch = nn.Linear(2, 1).cuda()

            # initialize all fc layers to xavier
            # TODO: Check if this initialization destroy the information from the pretrained conv layers
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, gain=1)

        def forward(self, images, explanations, scores):
            # Process the input images
            input_spatial_feats = self.conv_layers(images)
            input_feat = self.pooling_layer(input_spatial_feats).squeeze()
            input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsxcx49

            # Process the nearest neighbors
            explanations = explanations.squeeze()
            if len(explanations.shape) == 3:
                explanations = explanations.unsqueeze(0)
            explanation_spatial_feats = self.conv_layers(explanations)
            explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)


            # change from 2048x49 -> 49x2048
            # 49 tokens for an image
            input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)
            sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

            transformer_encoder_depth = RunningParams.L

            # Add the cls token and positional embedding --> 50x2048
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 1)

            # TODO: as the output of the first layer does not propagate to the second layer, then transformer_encoder_depth should be 1
            for _ in range(transformer_encoder_depth):
                # Self-attention --> 50x2048; both cls and image tokens are transformed.

                if self.transformer is not None:
                    input_spt_feats = self.transformer(input_spatial_feats)
                    exp_spt_feats = self.transformer(explanation_spatial_feats)
                else:
                    input_spt_feats = input_spatial_feats
                    exp_spt_feats = explanation_spatial_feats

                if self.cross_transformer is not None:
                    # Cross-attention --> 50x2048; only the cls tokens are transformed. Image tokens are kept the same.
                    input_spatial_feats, explanation_spatial_feats, i2e_attn, e2i_attn = self.cross_transformer(input_spt_feats, exp_spt_feats)
                else:
                    input_spatial_feats = input_spt_feats
                    explanation_spatial_feats = exp_spt_feats

            input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
            input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

            # Extracting the cls token --> 1x2048
            input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))

            pairwise_feats = []

            x = self.branch3(
                torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1))

            output3 = x

            output3 = self.agg_branch(output3)

            output = output3

            return output, input_feat, None, None

    class CNN_AdvisingNetwork(nn.Module):
        def __init__(self):
            print("Using training network for Birds (CUB-200)")
            super(CNN_AdvisingNetwork, self).__init__()

            from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck
            resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')
            resnet.load_state_dict(my_model_state_dict, strict=True)

            conv_features = list(resnet.children())[:RunningParams.conv_layer-6]  # delete the last fc layer
            self.conv_layers = nn.Sequential(*conv_features)

            self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.branch3 = BinaryMLP(
                2 * RunningParams.conv_layer_size[RunningParams.conv_layer] + 2, 32)

            self.agg_branch = nn.Linear(2, 1).cuda()


            # initialize all fc layers to xavier
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, gain=1)

        def forward(self, images, explanations, scores):
            # Process the input images
            input_spatial_feats = self.conv_layers(images)
            input_feat = self.pooling_layer(input_spatial_feats).squeeze()

            # Process the nearest neighbors
            explanations = explanations.squeeze()
            explanation_spatial_feats = self.conv_layers(explanations)

            ################################
            exp_feat = self.pooling_layer(explanation_spatial_feats).squeeze()
            ################################

            sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

            ################################
            x = self.branch3(
                torch.cat([sep_token, input_feat, sep_token, exp_feat], dim=1))
            ################################

            output3 = self.agg_branch(x)

            output = output3

            return output, input_feat, None, None

    class ViT_AdvisingNetwork(nn.Module):
        def __init__(self):
            print("Using training ViT network for Birds (CUB-200)")
            super(ViT_AdvisingNetwork, self).__init__()

            import timm
            class CustomViT(nn.Module):
                def __init__(self, base_model):
                    super(CustomViT, self).__init__()
                    self.base_model = base_model

                def forward(self, x):
                    # Get the features from the base ViT model
                    x = self.base_model.forward_features(x)
                    # Extract the CLS token (first token)
                    cls_token = x[:, 0]
                    # Pass the features through the classifier
                    output = self.base_model.head(cls_token)
                    return output, cls_token

            # Initialize the base model and load the trained weights
            base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=200)
            model_path = "./vit_base_patch16_224_cub_200way.pth"
            state_dict = torch.load(model_path, map_location=torch.device("cuda"))
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            base_model.load_state_dict(new_state_dict)

            # Wrap the base model in the custom model
            self.feature_extractor = CustomViT(base_model)

            # 768 is the dimension of ViT-B-16
            self.branch3 = BinaryMLP(
                2 * 768 + 2, 32)

            self.agg_branch = nn.Linear(2, 1).cuda()

            # initialize branch3 and agg_branch only
            for m in self.modules():
                if isinstance(m, nn.Linear) and not isinstance(m, CustomViT):
                    torch.nn.init.xavier_normal_(m.weight, gain=1)

        def forward(self, images, explanations, scores):
            # Process the input images
            _, input_cls = self.feature_extractor(images)

            # Process the nearest neighbors
            explanations = explanations.squeeze()
            _, explanation_cls = self.feature_extractor(explanations)

            sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

            ################################
            x = self.branch3(
                torch.cat([sep_token, input_cls, sep_token, explanation_cls], dim=1))
            ################################

            output3 = self.agg_branch(x)

            output = output3

            return output, None, None, None

elif RunningParams.CARS_TRAINING is True:
    class Transformer_AdvisingNetwork(nn.Module):
        def __init__(self):
            print("Using training network for Cars")
            super(Transformer_AdvisingNetwork, self).__init__()

            ################################################################
            import torchvision

            if RunningParams.resnet == 50:
                model = torchvision.models.resnet50(pretrained=True).cuda()
            elif RunningParams.resnet == 34:
                model = torchvision.models.resnet34(pretrained=True).cuda()
            elif RunningParams.resnet == 18:
                model = torchvision.models.resnet18(pretrained=True).cuda()
            model.fc = nn.Linear(model.fc.in_features, 196)

            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cars-196/model_best_rn{RunningParams.resnet}.pth.tar', map_location=torch.device('cpu'))
            model.load_state_dict(my_model_state_dict['state_dict'], strict=True)
            ################################################################

            conv_features = list(model.children())[:RunningParams.conv_layer - 6]  # delete the last fc layer
            self.conv_layers = nn.Sequential(*conv_features)

            self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            class transformer_feat_embedder(nn.Module):
                def __init__(self, num_patches, dim):
                    super().__init__()
                    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
                    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

                def forward(self, feat, k_value):
                    b, n, _ = feat.shape
                    cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
                    x = torch.cat((cls_tokens, feat), dim=1)
                    x += self.pos_embedding[:, :(n + 1)]

                    return x

            self.transformer_feat_embedder = transformer_feat_embedder(
                RunningParams.feat_map_size[RunningParams.conv_layer],
                RunningParams.conv_layer_size[RunningParams.conv_layer])

            transformer_depth = RunningParams.N
            cross_transformer_depth = RunningParams.M
            feat_dim = RunningParams.conv_layer_size[RunningParams.conv_layer]
            self.transformer = Transformer(dim=feat_dim, depth=transformer_depth, heads=8, dim_head=64, mlp_dim=512,
                                           dropout=0.0)
            self.cross_transformer = CrossTransformer(sm_dim=feat_dim, lg_dim=feat_dim, depth=cross_transformer_depth,
                                                      heads=8,
                                                      dim_head=64, dropout=0.0)

            print('Using transformer with depth {} and 8 heads'.format(transformer_depth))
            print('Using cross_transformer with depth {} and 8 heads'.format(cross_transformer_depth))

            self.branch3 = BinaryMLP(
                2 * RunningParams.conv_layer_size[RunningParams.conv_layer] + 2, 32)

            self.agg_branch = nn.Linear(2, 1).cuda()

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

            explanations = explanations.squeeze()
            explanation_spatial_feats = self.conv_layers(explanations)
            explanation_spatial_feats = explanation_spatial_feats.flatten(start_dim=2)

            # change from 2048x49 -> 49x2048
            # 49 tokens for an image
            input_spatial_feats = torch.transpose(input_spatial_feats, 1, 2)
            explanation_spatial_feats = torch.transpose(explanation_spatial_feats, 1, 2)

            sep_token = torch.zeros([explanations.shape[0], 1], requires_grad=False).cuda()

            transformer_encoder_depth = RunningParams.L

            # Add the cls token and positional embedding --> 50x2048
            input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats, 1)
            explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats, 1)

            # TODO: as the output of the first layer does not propagate to the second layer, then transformer_encoder_depth should be 1
            for _ in range(transformer_encoder_depth):
                # Self-attention --> 50x2048; both cls and image tokens are transformed.
                input_spt_feats = self.transformer(input_spatial_feats)
                exp_spt_feats = self.transformer(explanation_spatial_feats)

                # Cross-attention --> 50x2048; only the cls tokens are transformed. Image tokens are kept the same.
                input_spatial_feats, explanation_spatial_feats, i2e_attn, e2i_attn = self.cross_transformer(
                    input_spt_feats, exp_spt_feats)

            input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
            input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

            # Extracting the cls token --> 1x2048
            input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))

            pairwise_feats = []

            x = self.branch3(
                torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1))

            output3 = x

            output3 = self.agg_branch(output3)

            output = output3

            return output, input_feat, None, None

elif RunningParams.DOGS_TRAINING is True:
    class Transformer_AdvisingNetwork(nn.Module):
        def __init__(self):
            print("Using training network for Stanford Dogs")
            super(Transformer_AdvisingNetwork, self).__init__()

            ################################################################
            import torchvision

            if RunningParams.resnet == 50:
                model = torchvision.models.resnet50(pretrained=True).cuda()
                model.fc = nn.Linear(2048, 120)
                model.fc = nn.Sequential(nn.Linear(2048, 120)).cuda()
            elif RunningParams.resnet == 34:
                model = torchvision.models.resnet34(pretrained=True).cuda()
                model.fc = nn.Linear(512, 120)
            elif RunningParams.resnet == 18:
                model = torchvision.models.resnet18(pretrained=True).cuda()
                model.fc = nn.Linear(512, 120)

            print(f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth')
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth',
                map_location='cuda'
            )
            new_state_dict = {k.replace("model.", ""): v for k, v in my_model_state_dict.items()}

            ################################################################

            conv_features = list(model.children())[:RunningParams.conv_layer - 6]  # delete the last fc layer
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
                        cls_tokens = repeat(self.cls_token, '() n d -> b k n d', b=b, k=k)
                        x = torch.cat((cls_tokens, feat), dim=2)

                    x += self.pos_embedding[:, :(n + 1)]

                    return x

            self.transformer_feat_embedder = transformer_feat_embedder(
                RunningParams.feat_map_size[RunningParams.conv_layer],
                RunningParams.conv_layer_size[RunningParams.conv_layer])

            transformer_depth = RunningParams.N
            cross_transformer_depth = RunningParams.M
            feat_dim = RunningParams.conv_layer_size[RunningParams.conv_layer]
            self.transformer = Transformer(dim=feat_dim, depth=transformer_depth, heads=8, dim_head=64, mlp_dim=512,
                                           dropout=0.0)
            self.cross_transformer = CrossTransformer(sm_dim=feat_dim, lg_dim=feat_dim, depth=cross_transformer_depth,
                                                      heads=8,
                                                      dim_head=64, dropout=0.0)

            print('Using transformer with depth {} and 8 heads'.format(transformer_depth))
            print('Using cross_transformer with depth {} and 8 heads'.format(cross_transformer_depth))

            self.branch3 = BinaryMLP(
                2 * RunningParams.conv_layer_size[RunningParams.conv_layer] + 2, 32)

            if RunningParams.k_value == 1:
                self.agg_branch = nn.Linear(2, 1).cuda()
            else:
                self.agg_branch = nn.Linear(RunningParams.k_value*2, 1).cuda()

            #TODO?? Only initialize branch3 and agg_branch
            # initialize all fc layers to xavier
            # for m in self.modules():
            #     if isinstance(m, nn.Linear):
            #         torch.nn.init.xavier_normal_(m.weight, gain=1)

            # initialize branch3 and agg_branch only with Xavier initialization
            for m in self.modules():
                # Check if the module is a Linear layer and not in conv_layers
                if isinstance(m, nn.Linear) and m not in self.conv_layers.modules():
                    torch.nn.init.xavier_normal_(m.weight, gain=1)

        def forward(self, images, explanations, scores):
            # Process the input images
            input_spatial_feats = self.conv_layers(images)
            input_feat = self.pooling_layer(input_spatial_feats).squeeze()
            input_spatial_feats = input_spatial_feats.flatten(start_dim=2)  # bsxcx49

            # Process the nearest neighbors
            if RunningParams.k_value == 1:
                explanations = explanations.squeeze()
                explanation_spatial_feats = self.conv_layers(explanations)
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

            transformer_encoder_depth = RunningParams.L

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
                    input_spatial_feats, explanation_spatial_feats, i2e_attn, e2i_attn = self.cross_transformer(
                        input_spt_feats, exp_spt_feats)

                input_emb, exp_emb = input_spatial_feats, explanation_spatial_feats
                input_emb, exp_emb = input_emb.squeeze(), exp_emb.squeeze()

                # Extracting the cls token --> 1x2048
                input_cls, exp_cls = map(lambda t: t[:, 0], (input_emb, exp_emb))

            else:
                # Clone the input tensor K times along the second dimension
                input_spatial_feats = input_spatial_feats.unsqueeze(1).repeat(1, RunningParams.k_value, 1, 1). \
                    view(input_spatial_feats.shape[0], RunningParams.k_value, 49,
                         RunningParams.conv_layer_size[RunningParams.conv_layer])
                input_spatial_feats = self.transformer_feat_embedder(input_spatial_feats,
                                                                     RunningParams.k_value)  # bsx50x2048
                explanation_spatial_feats = self.transformer_feat_embedder(explanation_spatial_feats,
                                                                           RunningParams.k_value)  # bsxKx50x2048

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

                input_emb = input_spatial_feats
                input_emb = input_emb.squeeze()
                if RunningParams.k_value == 1:
                    input_cls = input_emb[:, 0]
                else:
                    input_cls = input_emb[:, :, 0]

                exp_emb = explanation_spatial_feats
                exp_emb = exp_emb.squeeze()
                if RunningParams.k_value == 1:
                    exp_cls = exp_emb[:, 0]
                else:
                    exp_cls = exp_emb[:, :, 0]

                i2e_attns = torch.cat(i2e_attns, dim=2)
                e2i_attns = torch.cat(e2i_attns, dim=2)

            pairwise_feats = []

            if RunningParams.k_value == 1:
                x = self.branch3(
                    torch.cat([sep_token, input_cls, sep_token, exp_cls], dim=1))

                output3 = x
            else:
                for prototype_idx in range(0, RunningParams.k_value):
                    x = self.branch3(
                        torch.cat(
                            [sep_token, input_cls[:, prototype_idx], sep_token, exp_cls[:, prototype_idx]],
                            dim=1))
                    pairwise_feats.append(x)

                output3 = torch.cat(pairwise_feats, dim=1)

            output3 = self.agg_branch(output3)

            output = output3

            if RunningParams.k_value == 1:
                return output, input_feat, None, None
            else:
                return output, input_feat, i2e_attns, e2i_attns

else:
    print('Failed creating the model! Exiting...')
    exit(-1)