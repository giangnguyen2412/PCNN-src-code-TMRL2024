import models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
from helpers import HelperFunctions
from models import MyCustomResnet18


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='', help='Model check point')
    parser.add_argument('--eval_dataset', type=str, default='', help='Evaluation dataset')

    args = parser.parse_args()
    model_path = args.ckpt
    print(args)

    model = MyCustomResnet18(pretrained=False, fine_tune=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    model.eval()


