import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import faiss
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from models import AdvisingNetwork
from params import RunningParams
from datasets import Dataset, ImageFolderForZeroshot
from torchvision.datasets import ImageFolder
# from visualize import Visualization
from helpers import HelperFunctions
from explainers import ModelExplainer
from transformer import Transformer_AdvisingNetwork
from visualize import Visualization

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
ModelExplainer = ModelExplainer()
Visualization = Visualization()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_eager-pine-2791.pt',
                        # default='best_model_apricot-paper-2768.pt',
                        # default='best_model_hopeful-cloud-2789.pt',
                        default='best_model_olive-field-2793.pt',
                        # default='best_model_rosy-violet-2795.pt',
                        # default='best_model_hopeful-totem-2790.pt',
                        # default='best_model_fragrant-sea-2785.pt',
                        # default='best_model_ancient-plant-2777.pt',
                        # default='best_model_eternal-dawn-2771.pt',
                        # default='best_model_fragrant-moon-2605.pt',
                        # default='best_model_wild-water-2279.pt',
                        # default='best_model_autumn-rain-1993.pt',
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join('best_models', args.ckpt)
    print(args)

    model = Transformer_AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    print('Validation accuracy: {:.2f}'.format(acc))

    model.eval()

    test_dir = '/home/giang/Downloads/nabirds_split_small/test'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForZeroshot(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=16,
            shuffle=False,  # turn shuffle to False
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Do not remove drop last because it affects performance
        )

        running_corrects = 0
        running_corrects_top5 = 0
        total_cnt = 0

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()

        infer_result_dict = dict()

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data[0].cuda()

            # Make a dummy confidence score
            model1_score = torch.rand([data[1].shape[0], 1]).cuda()

            output_tensors = []
            for class_idx in range(data[1].shape[1]):
                explanation = data[1][:, class_idx, :, :, :, :]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                output, _, _, _ = model(images=x, explanations=explanation, scores=model1_score)
                output = output.squeeze()
                output_tensors.append(output)

            logits = torch.stack(output_tensors, dim=1)
            # convert logits to probabilities using softmax function
            p = torch.softmax(logits, dim=0)

            # Compute top-1 predictions and accuracy
            score, index = torch.topk(p, 1, dim=1)
            running_corrects += torch.sum(index.squeeze() == gt.cuda())

            # Compute top-5 predictions and accuracy
            score_top5, index_top5 = torch.topk(p, 5, dim=1)
            gt_expanded = gt.cuda().unsqueeze(1).expand_as(index_top5)
            running_corrects_top5 += torch.sum(index_top5 == gt_expanded)

            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects*100/total_cnt))
            print("Top-5 Accuracy: {}".format(running_corrects_top5*100/total_cnt))

