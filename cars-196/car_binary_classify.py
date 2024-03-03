import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from transformer import Transformer_AdvisingNetwork
from visualize import Visualization
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from torchvision import datasets, models, transforms

RunningParams = RunningParams('CARS')

Dataset = Dataset()
HelperFunctions = HelperFunctions()
Visualization = Visualization()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

CATEGORY_ANALYSIS = False

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/{RunningParams.train_path}',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_' + RunningParams.wandb_sess_name + '.pt',
                        default='best_model_robust-sunset-3158.pt',  # RN50 run 1, 94.97
                        # default='best_model_wandering-capybara-3189.pt',  # RN50 run 2, 95.07
                        # default='best_model_different-lion-3192.pt',  # RN50 run 3, 95.04

                        # default='best_model_spring-field-3157.pt',  # RN34
                        # default='best_model_divine-cherry-3160.pt',  # RN18
                        # default='best_model_summer-waterfall-3168.pt',  # RN50 not data augmentation, 93.50
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)
    print(args)

    MODEL2 = Transformer_AdvisingNetwork()
    MODEL2 = nn.DataParallel(MODEL2).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']

    MODEL2.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    print('Validation accuracy: {:.2f}'.format(acc))

    print(RunningParams.__dict__)

    MODEL2.eval()

    ################################################################

    import torchvision

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
    model.eval()

    MODEL1 = model.cuda()
    MODEL1.eval()

    in_features = model.fc.in_features

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                         ])

    ################################################################

    # test_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'
    test_dir = RunningParams.aug_data_dir

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, data_transform)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    ################################################################
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=64,
            shuffle=True,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
            drop_last=False  # Do not remove drop last because it affects performance
        )

        running_corrects = 0
        advising_crt_cnt = 0

        top1_cnt, top1_crt_cnt = 0, 0

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()

        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']
        save_dir = f'{RunningParams.prj_dir}/vis/cars'
        if RunningParams.VISUALIZE_COMPARATOR_CORRECTNESS is True:
            # HelperFunctions.check_and_rm(save_dir)
            HelperFunctions.check_and_mkdir(save_dir)
        model1_confidence_dist = dict()
        model2_confidence_dist = dict()
        for cat in categories:
            model1_confidence_dist[cat] = list()
            model2_confidence_dist[cat] = list()
            HelperFunctions.check_and_mkdir(os.path.join(save_dir, cat))

        infer_result_dict = dict()

        labels_val = []
        preds_val = []

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data[0].cuda()

            if len(data_loader.dataset.classes) < 196:
                for sample_idx in range(x.shape[0]):
                    tgt = gt[sample_idx].item()
                    class_name = data_loader.dataset.classes[tgt]
                    id = full_cub_dataset.class_to_idx[class_name]
                    gt[sample_idx] = id

            gts = gt.cuda()
            # Step 1: Forward pass input x through MODEL1 - Explainer
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            model1_score, index = torch.topk(model1_p, 1, dim=1)
            predicted_ids = index.squeeze()
            # MODEL1 Y/N label for input x
            for sample_idx in range(x.shape[0]):
                query = pths[sample_idx]
                base_name = os.path.basename(query)

            # model2_gt = (predicted_ids == gts) * 1  # 0 and 1
            # labels = model2_gt
            labels = data[2].cuda()

            # Get the idx of wrong predictions
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]

            if 'train' in test_dir or 'val' in test_dir:
                labels = data[2].cuda()

            # Generate explanations
            explanation = data[1]
            explanation = explanation[:, 0:RunningParams.k_value, :, :, :]
            # Find the maximum value along each row
            max_values, _ = torch.max(model1_p, dim=1)

            SAME = False
            RAND = False

            if SAME is True:
                model1_score.fill_(0.999)

                explanation = x.clone().unsqueeze(1).repeat(1, RunningParams.k_value, 1, 1, 1)
            if RAND is True:
                torch.manual_seed(42)
                explanation = torch.rand_like(explanation)
                explanation.cuda()
                # Replace the maximum value with random guess
                model1_score.fill_(1 / 196)

            # explanation = explanation[torch.randperm(explanation.size(0))]

            # Forward input, explanations, and softmax scores through MODEL2
            output, query, i2e_attn, e2i_attn = MODEL2(images=x, explanations=explanation, scores=model1_score)

            # convert logits to probabilities using sigmoid function
            p = torch.sigmoid(output)

            # classify inputs as 0 or 1 based on the threshold of 0.5
            preds = (p >= 0.5).long().squeeze()
            model2_score = p

            results = (preds == labels)

            preds_val.append(preds)
            labels_val.append(labels)

            running_corrects += torch.sum(preds == labels.data)

            for j in range(x.shape[0]):
                pth = pths[j]
                VISUALIZE_COMPARATOR_HEATMAPS = False
                if VISUALIZE_COMPARATOR_HEATMAPS is True:
                    i2e_attn = i2e_attn.mean(dim=1)  # bsx8x3x50
                    i2e_attn = i2e_attn[:, :, 1:]  # remove cls token --> bsx3x49

                    e2i_attn = e2i_attn.mean(dim=1)
                    e2i_attn = e2i_attn[:, :, 1:]  # remove cls token

                    for sample_idx in range(x.shape[0]):
                        if sample_idx == 1:
                            break
                        result = results[sample_idx].item()
                        if result is True:
                            correctness = 'Correctly'
                        else:
                            correctness = 'Incorrectly'

                        pred = preds[sample_idx].item()
                        if pred == 1:
                            action = 'Accept'
                        else:
                            action = 'Reject'
                        model2_decision = correctness + action

                        query = pths[sample_idx]
                        base_name = os.path.basename(query)
                        prototypes = data_loader.dataset.faiss_nn_dict[base_name][0:RunningParams.k_value]
                        for prototype_idx in range(RunningParams.k_value):
                            bef_weights = i2e_attn[sample_idx, prototype_idx:prototype_idx + 1, :]
                            aft_weights = e2i_attn[sample_idx, prototype_idx:prototype_idx + 1, :]

                            # as the test images are from validation, then we do not need to exclude the fist prototype
                            prototype = prototypes[prototype_idx]
                            Visualization.visualize_transformer_attn_v2(bef_weights, aft_weights, prototype, query,
                                                                        model2_decision, prototype_idx)

                        # Combine visualizations
                        cmd = 'montage tmp/{}/{}_[0-{}].png -tile 3x1 -geometry +0+0 tmp/{}/{}.jpeg'.format(
                            model2_decision, base_name, RunningParams.k_value - 1, model2_decision, base_name)
                        # cmd = 'montage tmp/{}/{}_[0-{}].png -tile 3x1 -geometry +0+0 my_plot.png'.format(
                        #     model2_decision, base_name, RunningParams.k_value - 1)
                        os.system(cmd)
                        # Remove unused images
                        cmd = 'rm -rf tmp/{}/{}_[0-{}].png'.format(
                            model2_decision, base_name, RunningParams.k_value - 1)
                        os.system(cmd)

            if RunningParams.VISUALIZE_COMPARATOR_CORRECTNESS is True:
                for sample_idx in range(x.shape[0]):
                    result = results[sample_idx].item()
                    if result is True:
                        correctness = 'Correctly'
                    else:
                        correctness = 'Incorrectly'

                    pred = preds[sample_idx].item()
                    if pred == 1:
                        action = 'Accept'
                    else:
                        action = 'Reject'

                    model2_decision = correctness + action
                    query = pths[sample_idx]

                    # TODO: move this out to remove redundancy
                    base_name = os.path.basename(query)
                    save_path = os.path.join(save_dir, model2_decision, base_name)

                    gt_label = full_cub_dataset.classes[gts[sample_idx].item()]
                    pred_label = full_cub_dataset.classes[predicted_ids[sample_idx].item()]

                    # model1_confidence = int(model1_score[sample_idx].item() * 100)
                    # model2_confidence = int(model2_score[sample_idx].item() * 100)

                    model1_confidence = model1_score[sample_idx].item()
                    model2_confidence = model2_score[sample_idx].item()

                    model1_confidence_dist[model2_decision].append(model1_confidence)
                    model2_confidence_dist[model2_decision].append(model2_confidence)

                    # Finding the extreme cases
                    # if (action == 'Accept' and confidence < 50) or (action == 'Reject' and confidence > 80):
                    if correctness == 'Incorrectly':
                        prototypes = data_loader.dataset.faiss_nn_dict[base_name]['NNs'][0:RunningParams.k_value]
                        Visualization.visualize_model2_decision_with_prototypes(query,
                                                                                gt_label,
                                                                                pred_label,
                                                                                model2_decision,
                                                                                save_path,
                                                                                save_dir,
                                                                                model1_confidence,
                                                                                model2_confidence,
                                                                                prototypes)

                    infer_result_dict[base_name] = dict()
                    infer_result_dict[base_name]['model2_decision'] = model2_decision
                    infer_result_dict[base_name]['confidence1'] = model1_confidence
                    infer_result_dict[base_name]['confidence2'] = model2_confidence
                    infer_result_dict[base_name]['gt_label'] = gt_label
                    infer_result_dict[base_name]['pred_label'] = pred_label
                    infer_result_dict[base_name]['result'] = result is True

            yes_cnt += sum(preds)
            true_cnt += sum(labels)
            # np.save('infer_results/{}.npy'.format(args.ckpt), infer_result_dict)

        # breakpoint()
        epoch_acc = running_corrects.double() / len(image_datasets[ds])
        yes_ratio = yes_cnt.double() / len(image_datasets[ds])
        true_ratio = true_cnt.double() / len(image_datasets[ds])

        orig_wrong = len(image_datasets[ds]) - true_cnt
        adv_wrong = len(image_datasets[ds]) - advising_crt_cnt

        ################################################################
        # Calculate precision, recall, and F1 score
        preds_val = torch.cat(preds_val, dim=0)
        labels_val = torch.cat(labels_val, dim=0)

        precision = precision_score(labels_val.cpu(), preds_val.cpu())
        recall = recall_score(labels_val.cpu(), preds_val.cpu())
        f1 = f1_score(labels_val.cpu(), preds_val.cpu())
        confusion_matrix_ = confusion_matrix(labels_val.cpu(), preds_val.cpu())
        print(confusion_matrix_)

        print('Acc: {:.2f} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}'.format(
            epoch_acc.item() * 100, precision, recall, f1))
        ################################################################

        print('{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. accuracy: {:.2f}'.format(
            os.path.basename(test_dir), epoch_acc * 100, yes_ratio * 100, true_ratio * 100))

        np.save('confidence.npy', confidence_dict)
