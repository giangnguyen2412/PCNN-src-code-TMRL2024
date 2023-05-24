import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy
import wandb
import statistics
import pdb
import random
import numpy as np

from tqdm import tqdm
from torchvision import datasets, models, transforms
from transformer import *
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

RunningParams = RunningParams()
Dataset = Dataset()
Explainer = ModelExplainer()

assert (RunningParams.DOGS_TRAINING is False and RunningParams.IMAGENET_TRAINING is False)

if [RunningParams.IMAGENET_TRAINING, RunningParams.DOGS_TRAINING, RunningParams.CUB_TRAINING].count(True) > 1:
    print("There are more than one training datasets chosen, skipping training!!!")
    exit(-1)

if RunningParams.MODEL2_FINETUNING is True or RunningParams.UNBALANCED_TRAINING:
    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    MODEL1 = resnet.cuda()
    MODEL1.eval()
    fc = list(MODEL1.children())[-1].cuda()
    fc = nn.DataParallel(fc)
else:
    import torchvision

    inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
    inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
    my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
    inat_resnet.load_state_dict(my_model_state_dict, strict=True)
    MODEL1 = inat_resnet
    MODEL1.eval()

    fc = MODEL1.fc
    fc = fc.cuda()

if RunningParams.MODEL2_FINETUNING is True:
    train_dataset = '/home/giang/Downloads/Final_RN50_dataset_CUB_HP/final_train_aug'
    val_dataset = '/home/giang/Downloads/Final_RN50_dataset_CUB_HP/final_val'
    if RunningParams.UNBALANCED_TRAINING is True:
        # train_dataset = '/home/giang/Downloads/datasets/CUB_train'
        # train_dataset = '/home/giang/Downloads/datasets/CUB_train_all_backup2'
        train_dataset = '/home/giang/Downloads/datasets/CUB_train_all_top5'
        # train_dataset = '/home/giang/Downloads/datasets/CUB_train_aug'
        val_dataset = '/home/giang/Downloads/datasets/CUB_val'
else:
    train_dataset = '/home/giang/Downloads/datasets/CUB_pre_train'
    val_dataset = '/home/giang/Downloads/datasets/CUB_pre_val'


full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if RunningParams.XAI_method == RunningParams.NNs:
    image_datasets = dict()
    image_datasets['train'] = ImageFolderForNNs(train_dataset, Dataset.data_transforms['train'])
    image_datasets['val'] = ImageFolderForNNs(val_dataset, Dataset.data_transforms['val'])
else:
    pass
    # Not implemented


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)


def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                shuffle = True
                model.train()  # Training mode

                for param in MODEL2.parameters():
                    param.requires_grad = False

                if RunningParams.MODEL2_FINETUNING is True:
                    for param in MODEL2.module.transformer_feat_embedder.parameters():
                        param.requires_grad_(True)

                    for param in MODEL2.module.conv_layers.parameters():
                        param.requires_grad_(True)

                    for param in MODEL2.module.transformer.parameters():
                        param.requires_grad_(True)

                for param in MODEL2.module.cross_transformer.parameters():
                    param.requires_grad_(True)

                for param in MODEL2.module.branch3.parameters():
                    param.requires_grad_(True)

                if RunningParams.THREE_BRANCH is True:
                    # for param in MODEL2.module.quality_branch.parameters():
                    #     param.requires_grad_(True)

                    for param in MODEL2.module.softmax_branch.parameters():
                        param.requires_grad_(True)

                    for param in MODEL2.module.agg_branch.parameters():
                        param.requires_grad_(True)
            else:
                shuffle = False
                model.eval()  # Evaluation mode

            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=RunningParams.batch_size,
                shuffle=shuffle,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
            )

            running_loss = 0.0
            running_corrects = 0

            yes_cnt = 0
            true_cnt = 0

            for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
                if RunningParams.XAI_method == RunningParams.NNs:
                    x = data[0].cuda()
                else:
                    x = data.cuda()
                if len(data_loader.dataset.classes) < 200:
                    for sample_idx in range(x.shape[0]):
                        tgt = gt[sample_idx].item()
                        class_name = data_loader.dataset.classes[tgt]
                        id = full_cub_dataset.class_to_idx[class_name]
                        gt[sample_idx] = id

                gts = gt.cuda()

                embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18

                out = fc(embeddings)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                # MODEL1 Y/N label for input x
                model2_gt = (predicted_ids == gts) * 1  # 0 and 1
                labels = model2_gt

                if phase == 'train':
                    labels = data[2].cuda()

                #####################################################

                if RunningParams.XAI_method == RunningParams.GradCAM:
                    explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.PRECOMPUTED_NN is True:
                        explanation = data[1]

                        explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if RunningParams.XAI_method == RunningParams.NO_XAI:
                        output, _, _ = model(images=x, explanations=None, scores=model1_p)
                    else:
                        output, query, nns, emb_cos_sim = model(images=x, explanations=explanation, scores=score)

                    p = torch.sigmoid(output)

                    # classify inputs as 0 or 1 based on the threshold of 0.5
                    preds = (p >= 0.5).long().squeeze()

                    loss = loss_func(output.squeeze(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == labels.data)

                yes_cnt += sum(preds)
                true_cnt += sum(labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            yes_ratio = yes_cnt.double() / len(image_datasets[phase])
            true_ratio = true_cnt.double() / len(image_datasets[phase])

            if phase == 'train':
                scheduler.step()
                # scheduler.step(epoch_acc)

            wandb.log({'{}_accuracy'.format(phase): epoch_acc * 100, '{}_loss'.format(phase): epoch_loss})

            # print(memo)
            # print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
            #     wandb.run.name, phase, epoch_loss, epoch_acc * 100, yes_ratio * 100, true_ratio * 100))
            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc.item() * 100, yes_ratio.item() * 100, true_ratio.item() * 100))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                ckpt_path = '/home/giang/Downloads/advising_network/best_models/best_model_{}.pt' \
                    .format(wandb.run.name)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_loss,
                    'val_acc': epoch_acc * 100,
                    'val_yes_ratio': yes_ratio * 100,
                    'running_params': RunningParams,
                }, ckpt_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

MODEL2 = Transformer_AdvisingNetwork()

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

if RunningParams.CONTINUE_TRAINING:
    model_path = 'best_models/best_model_twilight-elevator-1955.pt'  # bottleneck false
    # model_path = 'best_models/best_model_legendary-lake-1956.pt'  # bottleneck false
    checkpoint = torch.load(model_path)
    MODEL2.load_state_dict(checkpoint['model_state_dict'])

    if RunningParams.EXP_TOKEN is True:
        MODEL2.module.branch3 = BinaryMLP(
            RunningParams.conv_layer_size[RunningParams.conv_layer] * 2 + 2, 32).cuda()
        MODEL2.module.agg_branch = nn.Linear(6, 1).cuda()

    ######################
    # initialize all fc layers to xavier
    for m in MODEL2.module.branch3.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1)

    for m in MODEL2.module.agg_branch.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1)
    ######################

    print('Continue training from ckpt {}'.format(model_path))
    print('Pretrained model accuracy: {:.2f}'.format(checkpoint['val_acc']))

if RunningParams.UNBALANCED_TRAINING is True:
    pos_weight = torch.tensor([4.0])  # the cost of misclassifying a positive sample
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

# Observe all parameters that are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

max_lr = RunningParams.learning_rate * 10
oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_ft, max_lr=max_lr,
    steps_per_epoch=dataset_sizes['train'] // RunningParams.batch_size,
    epochs=RunningParams.epochs)

config = {"train": train_dataset,
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'explanation': RunningParams.XAI_method,
          'k_value': RunningParams.k_value,
          'conv_layer': RunningParams.conv_layer,
          'USING_SOFTMAX': RunningParams.USING_SOFTMAX,
          'dropout_rate': RunningParams.dropout,
          'TOP1_NN': RunningParams.TOP1_NN,
          'MODEL2_FINETUNING': RunningParams.MODEL2_FINETUNING,
          'HIGHPERFORMANCE_FEATURE_EXTRACTOR': RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR,
          'HIGHPERFORMANCE_MODEL1': RunningParams.HIGHPERFORMANCE_MODEL1,
          'CONTINUE_TRAINING': RunningParams.CONTINUE_TRAINING,
          'BOTTLENECK': RunningParams.BOTTLENECK,
          'pos_w': RunningParams.pos_w,
          'THREE_BRANCH': RunningParams.THREE_BRANCH,
          'EXP_TOKEN': RunningParams.EXP_TOKEN,
          }

print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('datasets.py', policy='now')
wandb.save('cub_model_training.py', policy='now')
wandb.save('transformer.py', policy='now')

_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])

wandb.finish()