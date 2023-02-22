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
import statistics
import pdb
import random


from tqdm import tqdm
from torchvision import datasets, models, transforms
from models import AdvisingNetwork, TransformerAdvisingNetwork
from transformer import Transformer_AdvisingNetwork
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer

torch.backends.cudnn.benchmark = True
plt.ion()   # interactive mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,5,4"

RunningParams = RunningParams()
Dataset = Dataset()
Explainer = ModelExplainer()

action = RunningParams.action
if RunningParams.USING_CLASS_EMBEDDING is True:
    learnable = RunningParams.optim

commit = "Run k={} Dogs advising network with pretrained model RN50 bs{}. {} class embeddings.".format(
    RunningParams.k_value, RunningParams.batch_size, action
)

print(commit)
if len(commit) == 0:
    print("Please commit before running!")
    exit(-1)


if [RunningParams.IMAGENET_TRAINING, RunningParams.DOGS_TRAINING, RunningParams.CUB_TRAINING].count(True) > 1:
    print("There are more than one training datasets chosen, skipping training!!!")
    exit(-1)

assert (RunningParams.CUB_TRAINING is False)

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

if RunningParams.DOGS_TRAINING is True:
    MODEL1 = models.resnet34(pretrained=True).eval().cuda()
else:
    MODEL1 = models.resnet18(pretrained=True).eval().cuda()

fc = MODEL1.fc
fc = fc.cuda()

data_dir = '/home/giang/Downloads/advising_network'
# data_dir = '/home/giang/tmp'

virtual_train_dataset = '{}/train'.format(data_dir)
virtual_val_dataset = '{}/val'.format(data_dir)

# train_dataset = '/home/giang/Downloads/datasets/random_train_dataset'
train_dataset = '/home/giang/Downloads/datasets/balanced_train_dataset_180k'
val_dataset = '/home/giang/Downloads/datasets/balanced_val_dataset_6k'

TRAIN_DOG = RunningParams.DOGS_TRAINING
if TRAIN_DOG is True:
    train_dataset = '/home/giang/Downloads/datasets/Dogs_train'
    val_dataset = '/home/giang/Downloads/datasets/Dogs_val'

    category_val_dataset = '???'
    CATEGORY_ANALYSIS = False
    if CATEGORY_ANALYSIS is True:
        import glob

        correctness = ['Correct', 'Wrong']
        diffs = ['Easy', 'Medium', 'Hard']
        category_dict = {}
        category_record = {}

        for c in correctness:
            for d in diffs:
                dir = os.path.join(category_val_dataset, c, d)
                files = glob.glob(os.path.join(dir, '*', '*.*'))
                key = c + d
                for file in files:
                    base_name = os.path.basename(file)
                    category_dict[base_name] = key

                category_record[key] = {}
                category_record[key]['total'] = 0
                category_record[key]['crt'] = 0


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

imagenet_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/imagenet1k-val', Dataset.data_transforms['train'])

if RunningParams.XAI_method == RunningParams.NNs:
    image_datasets = {x: ImageFolderForNNs(os.path.join(data_dir, x), Dataset.data_transforms[x]) for x in ['train', 'val']}
else:
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), Dataset.data_transforms[x]) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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
                torch.set_grad_enabled(True)  # Using torch.no_grad without context manager
            else:
                shuffle = False
                model.eval()  # Evaluation mode
                torch.set_grad_enabled(False)

            data_loader = torch.utils.data.DataLoader(
                image_datasets[phase],
                batch_size=RunningParams.batch_size,
                shuffle=shuffle,  # turn shuffle to True
                num_workers=16,
                pin_memory=True,
                # drop_last=True
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
                gts = gt.cuda()

                embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18

                out = fc(embeddings)
                model1_p = torch.nn.functional.softmax(out, dim=1)
                score, index = torch.topk(model1_p, 1, dim=1)
                predicted_ids = index.squeeze()

                if TRAIN_DOG is True:
                    for sample_idx in range(x.shape[0]):
                        wnid = data_loader.dataset.classes[gts[sample_idx]]
                        gts[sample_idx] = imagenet_dataset.class_to_idx[wnid]

                        # wnid = imagenet_dataset.classes[predicted_ids[sample_idx]]
                        # predicted_ids[sample_idx] = data_loader.dataset.class_to_idx[wnid]

                # MODEL1 Y/N label for input x
                if RunningParams.IMAGENET_REAL and phase == 'val' and RunningParams.IMAGENET_TRAINING:
                    model2_gt = torch.zeros([x.shape[0]], dtype=torch.int64).cuda()
                    for sample_idx in range(x.shape[0]):
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)
                        real_ids = Dataset.real_labels[base_name]
                        if predicted_ids[sample_idx].item() in real_ids:
                            model2_gt[sample_idx] = 1
                        else:
                            model2_gt[sample_idx] = 0

                        if CATEGORY_ANALYSIS is True:
                            key = category_dict[base_name]
                            category_record[key]['total'] += 1
                else:
                    model2_gt = (predicted_ids == gts) * 1  # 0 and 1
                labels = model2_gt

                if RunningParams.USING_CLASS_EMBEDDING:
                    prototype_class_id_list = []
                    for sample_idx in range(x.shape[0]):
                        query_base_name = os.path.basename(pths[sample_idx])
                        prototype_list = data_loader.dataset.faiss_nn_dict[query_base_name]
                        if phase == 'train':
                            prototype_classes = [prototype_list[i].split('/')[-2] for i in
                                                 range(1, RunningParams.k_value + 1)]

                        else:
                            prototype_classes = [prototype_list[i].split('/')[-2] for i in range(0, RunningParams.k_value)]
                        prototype_class_ids = [imagenet_dataset.class_to_idx[c] for c in prototype_classes]
                        prototype_class_id_list.append(prototype_class_ids)

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
                            # black out the explanations
                            # explanation = torch.zeros_like(explanation)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if RunningParams.XAI_method == RunningParams.NO_XAI:
                        output, _, _ = model(images=x, explanations=None, scores=model1_p, prototype_class_id_list=prototype_class_id_list)
                    else:
                        output, query, _, _ = model(images=x, explanations=explanation, scores=model1_p, prototype_class_id_list=prototype_class_id_list)

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

                    if phase == 'val':
                        if CATEGORY_ANALYSIS is True:
                            for sample_idx in range(x.shape[0]):
                                query = pths[sample_idx]
                                base_name = os.path.basename(query)

                                key = category_dict[base_name]
                                if preds[sample_idx].item() == labels[sample_idx].item():
                                    category_record[key]['crt'] += 1

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

            if CATEGORY_ANALYSIS is True and phase == 'val':
                bin_acc = []
                for c in correctness:
                    for d in diffs:
                        print("{} - {} - {:.2f}".format(c, d,
                                                        category_record[c + d]['crt'] * 100 / category_record[c + d][
                                                            'total']))
                        bin_acc.append(category_record[c + d]['crt']/ category_record[c + d]['total'])
                avg_acc = statistics.mean(bin_acc)
                print('Avg acc - {:.2f}'.format(avg_acc))
                epoch_acc = avg_acc

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


MODEL2 = Transformer_AdvisingNetwork()

MODEL2 = MODEL2.cuda()
MODEL2 = nn.DataParallel(MODEL2)

if RunningParams.CONTINUE_TRAINING:
    model_path = 'best_models/best_model_whole-universe-848.pt'
    checkpoint = torch.load(model_path)
    MODEL2.load_state_dict(checkpoint['model_state_dict'])
    print('Continue training from ckpt {}'.format(model_path))
    print('Pretrained model accuracy: {:.2f}'.format(checkpoint['val_acc']))

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
    config=config,
    notes=commit,
)

wandb.save(os.path.basename(__file__), policy='now')
wandb.save('params.py', policy='now')
wandb.save('explainers.py', policy='now')
wandb.save('models.py', policy='now')
wandb.save('datasets.py', policy='now')
wandb.save('avs_traing.py', policy='now')
wandb.save('transformer.py', policy='now')
wandb.save('cross_vit.py', policy='now')


_, best_acc = train_model(
    MODEL2,
    criterion,
    optimizer_ft,
    oneLR_scheduler,
    config["num_epochs"])


wandb.finish()