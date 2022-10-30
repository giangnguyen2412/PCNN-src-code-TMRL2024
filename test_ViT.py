import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy
import wandb
import random
import pdb
import faiss

import torch
from cross_vit import CrossViT


from tqdm import tqdm
from torchray.attribution.grad_cam import grad_cam
from torchvision import datasets, models, transforms
from models import AdvisingNetwork
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

import torchvision as tv

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

os.environ['WANDB_MODE'] = 'offline'

RunningParams = RunningParams()
Dataset = Dataset()
Explainer = ModelExplainer()

model1_name = 'resnet18'
MODEL1 = models.resnet18(pretrained=True).eval()
fc = MODEL1.fc
fc = fc.cuda()

data_dir = '/home/giang/Downloads/advising_net_training/'
# data_dir = '/home/giang/tmp/'

virtual_train_dataset = '{}/train'.format(data_dir)

# train_dataset = '/home/giang/Downloads/datasets/random_train_dataset_1k'
train_dataset = '/home/giang/Downloads/datasets/balanced_train_dataset_60k'
val_dataset = '/home/giang/Downloads/datasets/balanced_val_dataset_6k'

if not HelperFunctions.is_program_running(os.path.basename(__file__)):
# if True:
    print('Creating symlink datasets...')
    if os.path.islink(virtual_train_dataset) is True:
        os.unlink(virtual_train_dataset)
    os.symlink(train_dataset, virtual_train_dataset)

    virtual_val_dataset = '{}/val'.format(data_dir)
    if os.path.islink(virtual_val_dataset) is True:
        os.unlink(virtual_val_dataset)
    os.symlink(val_dataset, virtual_val_dataset)
else:
    print('Script is running! No creating symlink datasets!')
if RunningParams.XAI_method == RunningParams.NNs:
    image_datasets = {x: ImageFolderForNNs(os.path.join(data_dir, x), Dataset.data_transforms[x]) for x in ['train', 'val']}
else:
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), Dataset.data_transforms[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
l1_dist = nn.PairwiseDistance(p=1)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

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

            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=RunningParams.batch_size,
                shuffle=True,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
            )

            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()  # Evaluation mode

            running_loss = 0.0
            running_corrects = 0

            running_label_loss = 0.0
            running_embedding_loss = 0.0

            yes_cnt = 0
            true_cnt = 0

            sim_0s = []
            sim_1s = []
            for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
                if RunningParams.XAI_method == RunningParams.NNs:
                    x = data[0].cuda()
                else:
                    x = data.cuda()
                gts = gt.cuda()

                embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18

                out = fc(embeddings)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                # MODEL1 Y/N label for input x
                if RunningParams.IMAGENET_REAL and phase == 'val':
                    model2_gt = torch.zeros([x.shape[0]], dtype=torch.int64).cuda()
                    for sample_idx in range(x.shape[0]):
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)
                        real_ids = Dataset.real_labels[base_name]
                        if predicted_ids[sample_idx].item() in real_ids:
                            model2_gt[sample_idx] = 1
                        else:
                            model2_gt[sample_idx] = 0
                else:
                    model2_gt = (predicted_ids == gts) * 1  # 0 and 1
                labels = model2_gt

                if RunningParams.XAI_method == RunningParams.GradCAM:
                    explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
                elif RunningParams.XAI_method == RunningParams.NNs:
                    if RunningParams.PRECOMPUTED_NN is True:
                        explanation = data[1]
                        if phase == 'train':
                            explanation = explanation[:, 1:RunningParams.k_value + 1, :, :, :]  # ignore 1st NN = query
                        else:
                            explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if RunningParams.XAI_method == RunningParams.NO_XAI:
                        pass
                    elif RunningParams.XAI_method == RunningParams.NNs:
                        if RunningParams.k_value == 1:
                            output = model(x, explanation.squeeze())
                        else:
                            assert "Not implemented yet!"
                            return -1

                    p = torch.nn.functional.softmax(output, dim=1)
                    confs, preds = torch.max(p, 1)
                    label_loss = loss_func(p, labels)

                    if RunningParams.XAI_method == RunningParams.NNs and RunningParams.embedding_loss is True:
                        loss = label_loss + embedding_loss
                        # loss = embedding_loss
                    else:
                        loss = label_loss

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
                scheduler.step()
                # scheduler.step(epoch_acc)

            wandb.log({'{}_accuracy'.format(phase): epoch_acc, '{}_loss'.format(phase): epoch_loss})

            print('{} - Loss: {:.4f} - Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                phase, epoch_loss, epoch_acc*100, yes_ratio * 100, true_ratio*100))
            # if RunningParams.XAI_method == RunningParams.NNs:
            #     print('{} - Label Loss: {:.4f} - Embedding Loss: {:.4f} '.format(
            #         phase, epoch_label_loss, epoch_embedding_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                ckpt_path = '/home/giang/Downloads/advising_net_training/best_models/best_model_{}.pt'\
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


model2_name = 'CrossViT'
# MODEL2 = AdvisingNetwork()

MODEL2 = CrossViT(
    image_size = 224,  # Change from 256 to 224
    num_classes = 2,  # Change from 1000 to 2 (binary classification)
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 192,            # low res dimension: 384 -> 192 to make it same with above
    lg_patch_size = 16,      # low res patch size: 64 -> 16 to make it same with above
    lg_enc_depth = 2,        # low res depth: 3 -> 2 to make it same with above
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

if RunningParams.CONTINUE_TRAINING:
    model_path = 'best_models/best_model_pretty-puddle-331.pt'
    checkpoint = torch.load(model_path)
    MODEL2.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()

# Observe all parameters that are being optimized
optimizer_ft = optim.SGD(MODEL2.parameters(), lr=RunningParams.learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

oneLR_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_ft, max_lr=0.01,
    steps_per_epoch=dataset_sizes['train']//RunningParams.batch_size,
    epochs=RunningParams.epochs)

# change to ReduceLROnPlateau scheduler, 'min': reduce LR if not decreasing
# oneLR_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=4)

config = {"train": train_dataset,
          "val": val_dataset,
          "train_size": dataset_sizes['train'],
          "val_size": dataset_sizes['val'],
          "model1": model1_name,
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
          'CrossCorrelation': RunningParams.CrossCorrelation,
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
wandb.save('models.py', policy='now')
wandb.save('datasets.py', policy='now')
wandb.save('train.py', policy='now')


_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])


wandb.finish()





# img1 = torch.randn(128, 3, 224, 224)  # 256 -> 224
# img2 = torch.randn(128, 3, 224, 224)
#
#
# pred = v(img1, img2)  # (1, 2) # 1000 -> 2
# print(pred)
# pass