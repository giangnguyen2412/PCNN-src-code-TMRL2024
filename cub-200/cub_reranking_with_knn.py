# Using cosine similarity to do reranking instead of using AdvNet
import torch
import torch.nn as nn
import os
import argparse
from torch.nn.functional import cosine_similarity

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams('CUB')

Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

torch.manual_seed(42)

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/{RunningParams.combined_path}',
                                     Dataset.data_transforms['train'])

from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)

    conv_features = list(resnet.children())[:RunningParams.conv_layer - 5]  # delete the last fc layer
    model = nn.Sequential(*conv_features)

    model = nn.DataParallel(model).cuda()

    # test_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/test'  ##################################
    test_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=17,
            shuffle=False,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
            drop_last=False  # Do not remove drop last because it affects performance
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
            labels = data[-1].cuda()

            if len(data_loader.dataset.classes) < 200:
                for sample_idx in range(x.shape[0]):
                    tgt = gt[sample_idx].item()
                    class_name = data_loader.dataset.classes[tgt]
                    id = full_cub_dataset.class_to_idx[class_name]
                    gt[sample_idx] = id

            output_tensors = []
            # Loop to get the logit for each class
            for class_idx in range(data[1].shape[1]):
                explanation = data[1][:, class_idx, :, :, :, :]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :].squeeze()

                x_conv = model(x).squeeze()
                ex_conv = model(explanation).squeeze()

                output = cosine_similarity(x_conv, ex_conv, dim=1)
                output_tensors.append(output)

            logits = torch.stack(output_tensors, dim=1)
            # convert logits to probabilities using softmax function
            p = torch.softmax(logits, dim=1)

            # Compute top-1 predictions and accuracy
            score, index = torch.topk(p, 1, dim=1)
            index = labels[torch.arange(len(index)), index.flatten()]

            running_corrects += torch.sum(index.squeeze() == gt.cuda())
            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))
