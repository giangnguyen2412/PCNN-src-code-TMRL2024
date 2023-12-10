def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from transformer import Transformer_AdvisingNetwork
import torch
import torch.nn as nn


model = Transformer_AdvisingNetwork()
model = model.cuda()
model = nn.DataParallel(model)

#
# model_path = 'best_models/best_model_decent-mountain-3215.pt'
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
#
print(count_parameters(model))

from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck
resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load(
    'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
resnet.load_state_dict(my_model_state_dict, strict=True)
print(count_parameters(resnet))

