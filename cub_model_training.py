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


from tqdm import tqdm
from torchvision import datasets, models, transforms
from models import AdvisingNetwork, TransformerAdvisingNetwork
from transformer import Transformer_AdvisingNetwork
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(0)

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6"

RunningParams = RunningParams()
Dataset = Dataset()
Explainer = ModelExplainer()

if [RunningParams.IMAGENET_TRAINING, RunningParams.DOGS_TRAINING, RunningParams.CUB_TRAINING].count(True) > 1:
    print("There are more than one training datasets chosen, skipping training!!!")
    exit(-1)

if RunningParams.MODEL2_FINETUNING is True:
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

data_dir = '/home/giang/Downloads/advising_network'
data_dir = '/home/giang/tmp'

virtual_train_dataset = '{}/train'.format(data_dir)
virtual_val_dataset = '{}/val'.format(data_dir)

if RunningParams.MODEL2_FINETUNING is True:
    train_dataset = '/home/giang/Downloads/RN50_dataset_CUB_HP_finetune_set/train'
    val_dataset = '/home/giang/Downloads/RN50_dataset_CUB_HP_finetune_set/val'
else:
    train_dataset = '/home/giang/Downloads/RN50_dataset_CUB_LP/train'
    val_dataset = '/home/giang/Downloads/RN50_dataset_CUB_LP/val'

if RunningParams.CUB_200WAY is True:
    train_dataset = '/home/giang/Downloads/datasets/CUB/train1/'
    val_dataset = '/home/giang/Downloads/datasets/CUB/test0/'

full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/CUB/combined',
                                         Dataset.data_transforms['train'])

if not HelperFunctions.is_program_running(os.path.basename(__file__)):
# if True:
    print('Creating symlink datasets...')
    if os.path.islink(virtual_train_dataset) is True:
        os.unlink(virtual_train_dataset)
    os.symlink(train_dataset, virtual_train_dataset)

    if os.path.islink(virtual_val_dataset) is True:
        os.unlink(virtual_val_dataset)
    os.symlink(val_dataset, virtual_val_dataset)
else:
    print('Script is running! No creating symlink datasets!')


if RunningParams.XAI_method == RunningParams.NNs:
    image_datasets = dict()
    image_datasets['train'] = ImageFolderForNNs(train_dataset, Dataset.data_transforms['train'])
    image_datasets['val'] = ImageFolderForNNs(val_dataset, Dataset.data_transforms['val'])
else:
    pass
    # Not implemented

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
l1_dist = nn.PairwiseDistance(p=1)

feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
feature_extractor.cuda()
feature_extractor = nn.DataParallel(feature_extractor)


def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                shuffle = True
                model.train()  # Training mode

                for param in MODEL2.parameters():
                    param.requires_grad = False

                for param in MODEL2.module.branch3.parameters():
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

            running_label_loss = 0.0
            running_embedding_loss = 0.0

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

                model2_gt = (predicted_ids == gts) * 1  # 0 and 1

                if RunningParams.CUB_200WAY is True:
                    labels = gts
                else:
                    labels = model2_gt

                if RunningParams.XAI_method == RunningParams.GradCAM:
                    explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.PRECOMPUTED_NN is True:
                        explanation = data[1]
                        if phase == 'train':
                            explanation = explanation[:, 1:RunningParams.k_value + 1, :, :, :]  # ignore 1st NN = query
                            # explanation = explanation[:, 2:, :, :, :]  # ignore 1st NN = query
                        else:
                            explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if RunningParams.XAI_method == RunningParams.NO_XAI:
                        output, _, _ = model(images=x, explanations=None, scores=model1_p)
                    else:
                        output, query, nns, emb_cos_sim = model(images=x, explanations=explanation, scores=model1_p)

                    p = torch.nn.functional.softmax(output, dim=1)
                    confs, preds = torch.max(p, 1)
                    label_loss = loss_func(p, labels)

                    loss = label_loss

                    # CONFIDENCE_LOSS = RunningParams.CONFIDENCE_LOSS
                    CONFIDENCE_LOSS = False
                    if CONFIDENCE_LOSS is True:
                        conf_loss = (1 - confs).mean()
                        loss = label_loss + conf_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_label_loss += label_loss.item() * x.size(0)
                # if RunningParams.XAI_method == RunningParams.NNs:
                #     running_embedding_loss += embedding_loss.item() * x.size(0)

                running_corrects += torch.sum(preds == labels.data)

                yes_cnt += sum(preds)
                true_cnt += sum(labels)

            import statistics
            # print("k: {} | Mean for True: {:.2f} and Mean for False: {:.2f}".format(RunningParams.k_value, statistics.mean(sim_1s), statistics.mean(sim_0s)))
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_label_loss = running_label_loss / len(image_datasets[phase])
            if RunningParams.XAI_method == RunningParams.NNs:
                epoch_embedding_loss = running_embedding_loss / len(image_datasets[phase])

            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            yes_ratio = yes_cnt.double() / len(image_datasets[phase])
            true_ratio = true_cnt.double() / len(image_datasets[phase])

            if phase == 'train':
                if RunningParams.CUB_200WAY is False:
                    scheduler.step()
                # scheduler.step(epoch_acc)

            wandb.log({'{}_accuracy'.format(phase): epoch_acc, '{}_loss'.format(phase): epoch_loss})

            print('{} - {} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                wandb.run.name, phase, epoch_loss, epoch_acc*100, yes_ratio * 100, true_ratio*100))
            # if RunningParams.XAI_method == RunningParams.NNs:
            #     print('{} - Label Loss: {:.4f} - Embedding Loss: {:.4f} '.format(
            #         phase, epoch_label_loss, epoch_embedding_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                ckpt_path = '/home/giang/Downloads/advising_network/best_models/best_model_{}.pt'\
                    .format(wandb.run.name)

                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_loss,
                    'val_acc': epoch_acc*100,
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

model2_name = 'Transformer_AdvisingNetwork'
MODEL2 = Transformer_AdvisingNetwork()

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

if RunningParams.CONTINUE_TRAINING:
    model_path = 'best_models/best_model_likely-bird-1000.pt'
    checkpoint = torch.load(model_path)
    MODEL2.load_state_dict(checkpoint['model_state_dict'])

    print('Continue training from ckpt {}'.format(model_path))
    print('Pretrained model accuracy: {:.2f}'.format(checkpoint['val_acc']))

criterion = nn.CrossEntropyLoss()

# Observe all parameters that are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

if RunningParams.CUB_200WAY is True:
    optimizer_ft = optim.Adam(MODEL2.parameters())

oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_ft, max_lr=0.01,
    steps_per_epoch=dataset_sizes['train']//RunningParams.batch_size,
    epochs=RunningParams.epochs)

config = {"train": train_dataset,
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "model2": model2_name,
          "num_epochs": RunningParams.epochs,
          "batch_size": RunningParams.batch_size,
          "learning_rate": RunningParams.learning_rate,
          'explanation': RunningParams.XAI_method,
          'query_frozen': RunningParams.query_frozen,
          'heatmap_frozen': RunningParams.heatmap_frozen,
          'nns_frozen':RunningParams.nns_frozen,
          'k_value': RunningParams.k_value,
          'ImageNetReaL': RunningParams.IMAGENET_REAL,
          'conv_layer': RunningParams.conv_layer,
          'embedding_loss': RunningParams.embedding_loss,
          'continue_training': RunningParams.CONTINUE_TRAINING,
          'using_softmax': RunningParams.USING_SOFTMAX,
          'dropout_rate': RunningParams.dropout,
          'TOP1_NN': RunningParams.TOP1_NN,
          'SIMCLR_MODEL': RunningParams.SIMCLR_MODEL,
          'COSINE_ONLY': RunningParams.COSINE_ONLY,
          }

print(config)
wandb.init(
    project="advising-network",
    entity="luulinh90s",
    config=config
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('explainers.py', policy='now')
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