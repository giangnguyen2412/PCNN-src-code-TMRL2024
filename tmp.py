def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from transformer import Transformer_AdvisingNetwork
import torch
import torch.nn as nn


model = Transformer_AdvisingNetwork()
model = model.cuda()
model = nn.DataParallel(model)


model_path = 'best_models/best_model_classic-wind-1138.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

print(count_parameters(model))
