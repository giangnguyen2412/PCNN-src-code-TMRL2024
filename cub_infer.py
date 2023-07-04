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
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize

abm_transform = A.Compose(
    [A.Resize(height=256, width=256),
     A.CenterCrop(height=224, width=224),
     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ToTensorV2(),
     ],

    additional_targets={'image0': 'image', 'image1': 'image',
                        'image2': 'image', 'image3': 'image', 'image4': 'image'}
)

CATEGORY_ANALYSIS = False

if RunningParams.MODEL2_ADVISING is True:
    in_features = 2048
    print("Building FAISS index...! Training set is the knowledge base.")
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/datasets/CUB_train_all',
                                         transform=Dataset.data_transforms['train'])

    faiss_data_loader = torch.utils.data.DataLoader(
        faiss_dataset,
        batch_size=RunningParams.batch_size,
        shuffle=False,  # turn shuffle to True
        num_workers=16,  # Set to 0 as suggested by
        # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
        pin_memory=True,
    )

    # TODO: CHANGE THIS IF CUB_EXTRACT_FEATURE CHANGES THE INDEX FILE NAME
    INDEX_FILE = 'faiss/cub/NeurIPS22_faiss_CUB200_class_idx_dict_HP_extractor.npy'

    if os.path.exists(INDEX_FILE):
        print("FAISS class index exists!")
        faiss_nns_class_dict = np.load(INDEX_FILE, allow_pickle="False", ).item()
        targets = faiss_data_loader.dataset.targets
        faiss_data_loader_ids_dict = dict()
        faiss_loader_dict = dict()
        for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
            faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if
                                                    targets[x] == class_id]  # check this value
            class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
            class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
            faiss_loader_dict[class_id] = class_id_loader
    else:
        print("Not found the FAISS index for class")
        exit(-1)

full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_divine-snowflake-2832.pt',
                        # default='best_model_driven-morning-2993.pt',
                        # default='best_model_atomic-microwave-2996.pt',
                        default='best_model_lilac-thunder-3015.pt',
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

    if RunningParams.MODEL_ENSEMBLE is True:
        print('The best confidence score was: {}'.format(checkpoint['best_conf']))
        print('Validation accuracy: {:.2f}'.format(acc / 100))
    else:
        print('Validation accuracy: {:.2f}'.format(acc))

    print(RunningParams.__dict__)

    model.eval()
    # test_dir = '/home/giang/Downloads/datasets/CUB/advnet/val'  ##################################
    test_dir = '/home/giang/Downloads/datasets/CUB/advnet/test'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    HIGHPERFORMANCE_MODEL1 = RunningParams.HIGHPERFORMANCE_MODEL1
    if HIGHPERFORMANCE_MODEL1 is True:
        from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

        resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load(
            'pretrained_models/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        resnet.load_state_dict(my_model_state_dict, strict=True)
        MODEL1 = resnet.cuda()
        MODEL1.eval()

        categorized_path = '/home/giang/Downloads/RN50_dataset_CUB_HIGH/combined'
    else:
        import torchvision

        inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
        inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
        my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
        inat_resnet.load_state_dict(my_model_state_dict, strict=True)
        MODEL1 = inat_resnet
        MODEL1.eval()

        categorized_path = '/home/giang/Downloads/RN50_dataset_CUB_LOW/combined'

    feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
    feature_extractor.cuda()
    feature_extractor = nn.DataParallel(feature_extractor)

    if CATEGORY_ANALYSIS is True:
        import glob

        correctness_bins = ['Correct', 'Wrong']
        category_dict = {}
        category_record = {}

        for c in correctness_bins:
            dir = os.path.join(categorized_path, c)
            files = glob.glob(os.path.join(dir, '*', '*.*'))
            key = c
            for file in files:
                base_name = os.path.basename(file)
                category_dict[base_name] = key

            category_record[key] = {}
            category_record[key]['total'] = 0
            category_record[key]['crt'] = 0

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=16,
            shuffle=False,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
            drop_last=False  # Do not remove drop last because it affects performance
        )

        running_corrects = 0
        advising_crt_cnt = 0

        top1_cnt, top1_crt_cnt = 0, 0

        MODEL_BAGGING = True
        if MODEL_BAGGING is True:
            running_corrects_conf_dict = {key: 0 for key in range(0, 96, 5)}
            # Init by 0.001 to avoid diving by ZERO
            total_conf_dict = {key: 0.001 for key in range(0, 96, 5)}
            thresholding_accs = [85.83, 85.83, 85.83, 85.87, 86.01, 86.17, 86.47, 86.89, 87.47,
                                 88.15, 88.97, 90.05, 90.91, 92.01, 92.81, 93.62, 94.60, 95.52, 96.42, 97.68]
            thresholding_accs_dict = {key: thresholding_accs[i] for i, key in enumerate(range(0, 96, 5))}
            thresholding_ratios = [1.0000, 1.0000, 1.0000, 0.9991, 0.9967, 0.9933, 0.9888, 0.9819,
                                   0.9722, 0.9570, 0.9392, 0.9173, 0.8947, 0.8723, 0.8469, 0.8198, 0.7891, 0.7551,
                                   0.7090, 0.6108]
            thresholding_ratios_dict = {key: thresholding_ratios[i] for i, key in enumerate(range(0, 96, 5))}

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()

        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']
        save_dir = '/home/giang/Downloads/advising_network/vis'
        if RunningParams.M2_VISUALIZATION is True:
            HelperFunctions.check_and_rm(save_dir)
            HelperFunctions.check_and_mkdir(save_dir)
        model1_confidence_dist = dict()
        model2_confidence_dist = dict()
        for cat in categories:
            model1_confidence_dist[cat] = list()
            model2_confidence_dist[cat] = list()
            HelperFunctions.check_and_mkdir(os.path.join(save_dir, cat))

        infer_result_dict = dict()

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
            import pdb
            # Step 1: Forward pass input x through MODEL1 - Explainer
            embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            model1_score, index = torch.topk(model1_p, 1, dim=1)
            _, model1_ranks = torch.topk(model1_p, 200, dim=1)
            predicted_ids = index.squeeze()
            # pdb.set_trace()
            # MODEL1 Y/N label for input x
            for sample_idx in range(x.shape[0]):
                query = pths[sample_idx]
                base_name = os.path.basename(query)

                if CATEGORY_ANALYSIS is True:
                    key = category_dict[base_name]
                    category_record[key]['total'] += 1

            model2_gt = (predicted_ids == gts) * 1  # 0 and 1
            labels = model2_gt

            # print(labels.shape)

            # Get the idx of wrong predictions
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]

            if 'train' in test_dir:
                labels = data[2].cuda()

            # print(labels.shape)

            # Generate explanations
            if RunningParams.XAI_method == RunningParams.GradCAM:
                explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
            elif RunningParams.XAI_method == RunningParams.NNs:
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
                    explanation = torch.rand_like(explanation) * (
                                explanation.max() - explanation.min()) + explanation.min()
                    explanation.cuda()
                    # Replace the maximum value with random guess
                    model1_score.fill_(1 / 200)

            if RunningParams.advising_network is True:
                # Forward input, explanations, and softmax scores through MODEL2
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    output, _, _ = model(images=x, explanations=None, scores=model1_p)
                else:
                    output, query, i2e_attn, e2i_attn = model(images=x, explanations=explanation, scores=model1_score)
                    # output, query, nns, emb_cos_sim = model(images=data[-1].cuda(), explanations=explanation,
                    #                                         scores=model1_score)

                # convert logits to probabilities using sigmoid function
                p = torch.sigmoid(output)

                # classify inputs as 0 or 1 based on the threshold of 0.5
                preds = (p >= 0.5).long().squeeze()
                model2_score = p

                if MODEL_BAGGING is True:
                    ###############################
                    conf_list = []
                    confidences = model1_score
                    confidences = (confidences * 100).long()
                    my_dict = {}
                    for key in range(0, 96, 5):
                        # get the indices where the tensor is smaller than the key
                        # advising net handles hard cases
                        indices = (confidences < key).nonzero().squeeze().view(-1, 2).tolist()
                        # add the indices to the dictionary
                        my_dict[key] = [id[0] for id in indices]

                        total_conf_dict[key] += len(my_dict[key])
                        running_corrects_conf_dict[key] += torch.sum((preds == labels.data)[my_dict[key]])
                    ###############################

                optimal_T = 90
                ##################################
                conf_list = []
                confidences = model1_score
                # for j, confidence in enumerate(confidences):
                #     confidence = confidence.item() * 100
                #     if confidence >= optimal_T:
                #         preds[j] = 1
                ###############################

                ##################################
                conf_list = []
                confidences = model1_score

                for j, confidence in enumerate(confidences):
                    confidence = confidence.item() * 100
                    thres_conf = thresholding_accs[int(confidence/5)]  # the confidence of the thresholding agent
                    adv_net_conf = p[j].item()*100  # the confidence of the advising network
                    if adv_net_conf < 50:
                        adv_net_conf = 100 - adv_net_conf

                    # # the thresholding is more confident than the adv net
                    # if thres_conf >= adv_net_conf:
                    #     preds[j] = 1
                    # else:
                    #     pass
                ###############################

                results = (preds == labels)
                for j in range(x.shape[0]):
                    pth = pths[j]

                    if '_0_' in pth:
                        top1_cnt += 1

                        if results[j] == True:
                            top1_crt_cnt += 1

                VISUALIZE_TRANSFORMER_ATTN = False
                if VISUALIZE_TRANSFORMER_ATTN is True:
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

                if CATEGORY_ANALYSIS is True:
                    for sample_idx in range(x.shape[0]):
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)

                        key = category_dict[base_name]
                        if preds[sample_idx].item() == labels[sample_idx].item():
                            category_record[key]['crt'] += 1

                running_corrects += torch.sum(preds == labels.data)

                AI_DELEGATE = True
                if AI_DELEGATE is True:
                    for sample_idx in range(x.shape[0]):
                        result = results[sample_idx].item()
                        pred = preds[sample_idx].item()

                        s = int(model1_score[sample_idx].item() * 100)
                        if s not in confidence_dict:
                            confidence_dict[s] = [0, 0, 0]
                            # if model2_score[sample_idx].item() >= model1_score[sample_idx].item():
                            if True:
                                if result is True:
                                    confidence_dict[s][0] += 1
                                else:
                                    confidence_dict[s][1] += 1

                                if labels[sample_idx].item() == 1:
                                    confidence_dict[s][2] += 1

                        else:
                            # if model2_score[sample_idx].item() >= model1_score[sample_idx].item():
                            if True:
                                if result is True:
                                    confidence_dict[s][0] += 1
                                else:
                                    confidence_dict[s][1] += 1

                                if labels[sample_idx].item() == 1:
                                    confidence_dict[s][2] += 1

            if RunningParams.M2_VISUALIZATION is True:
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

                    model1_confidence = int(model1_score[sample_idx].item() * 100)
                    model2_confidence = int(model2_score[sample_idx].item() * 100)
                    model1_confidence_dist[model2_decision].append(model1_confidence)
                    model2_confidence_dist[model2_decision].append(model2_confidence)

                    # Finding the extreme cases
                    # if (action == 'Accept' and confidence < 50) or (action == 'Reject' and confidence > 80):
                    if correctness == 'Incorrectly':
                        prototypes = data_loader.dataset.faiss_nn_dict[base_name][0:RunningParams.k_value]
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
            np.save('infer_results/{}.npy'.format(args.ckpt), infer_result_dict)

        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/vis/IncorrectlyAccept/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/vis/IncorrectlyAccept/*.jpg'
        os.system(cmd)
        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/vis/IncorrectlyReject/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/vis/IncorrectlyReject/*.jpg'
        os.system(cmd)

        epoch_acc = running_corrects.double() / len(image_datasets[ds])
        yes_ratio = yes_cnt.double() / len(image_datasets[ds])
        true_ratio = true_cnt.double() / len(image_datasets[ds])

        orig_wrong = len(image_datasets[ds]) - true_cnt
        adv_wrong = len(image_datasets[ds]) - advising_crt_cnt

        if MODEL_BAGGING is True:
            adv_net_acc_dict = {}
            ensemble_acc_dict = {}
            for key in range(0, 96, 5):
                adv_net_acc_dict[key] = running_corrects_conf_dict[key] * 100 / total_conf_dict[key]
                ensemble_acc_dict[key] = thresholding_accs_dict[key] * thresholding_ratios_dict[key] + \
                                         adv_net_acc_dict[key] * (1.0 - thresholding_ratios_dict[key])

                print(
                    'Using bagging - Optimal threshold: {} - Ensemble Acc: {:.2f} - Yes Ratio: {:.2f} - True Ratio: {:.2f}'.format(
                        key, ensemble_acc_dict[key], yes_ratio.item() * 100, true_ratio.item() * 100))

        if RunningParams.MODEL2_ADVISING is True:
            advising_acc = advising_crt_cnt.double() / len(image_datasets[ds])

            print(
                '{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. Acc: {:.2f} - Correction Acc: {:.2f}'.format(
                    ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100, advising_acc * 100))
            print('Original misclassified: {} - After correction: {}'.format(orig_wrong, adv_wrong))
        else:
            print('{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. accuracy: {:.2f}'.format(
                ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100))

        if CATEGORY_ANALYSIS is True:
            print(category_record)
            for c in correctness_bins:
                print("{} - {:.2f}".format(c, category_record[c]['crt'] * 100 / category_record[c]['total']))

        np.save('confidence.npy', confidence_dict)

        # print('Top1 acc: {:.2f}'.format(top1_crt_cnt*100/top1_cnt))
