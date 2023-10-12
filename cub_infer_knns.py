import torch
import torch.nn as nn
import numpy as np
import os
import argparse

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderWithPaths, ImageFolderForNNs
from helpers import HelperFunctions
from transformer import Transformer_AdvisingNetwork
from visualize import Visualization
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
Visualization = Visualization()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

CATEGORY_ANALYSIS = False

full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_genial-plasma-3125.pt',
                        default='best_model_decent-pyramid-3156.pt',
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join('best_models', args.ckpt)
    print(args)

    MODEL2 = Transformer_AdvisingNetwork()
    MODEL2 = nn.DataParallel(MODEL2).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    MODEL2.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    print('Validation accuracy: {:.2f}'.format(acc))

    print(RunningParams.__dict__)

    MODEL2.eval()
    # test_dir = '/home/giang/Downloads/datasets/CUB/advnet/val'  ##################################
    # test_dir = '/home/giang/Downloads/datasets/CUB/advnet/test'  ##################################
    test_dir = '/home/giang/Downloads/datasets/CUB/test0'  ##################################

    image_datasets = dict()
    nn_num = 5
    file_name = 'faiss/cub/top{}_k{}_enriched_NeurIPS_Finetuning_faiss_test5k7_top1_HP_MODEL1_HP_FE.npy'.format(
        1, nn_num)

    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, Dataset.data_transforms['val'], nn_dict=file_name)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

    resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')
    resnet.load_state_dict(my_model_state_dict, strict=True)
    MODEL1 = resnet.cuda()
    MODEL1.eval()

    categorized_path = '/home/giang/Downloads/RN50_dataset_CUB_HIGH/combined'


    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=10,
            shuffle=False,  # turn shuffle to False
            num_workers=8,
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

        labels_val = []
        preds_val = []

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

            model2_gt = (predicted_ids == gts) * 1  # 0 and 1
            labels = model2_gt

            # print(labels.shape)

            # Get the idx of wrong predictions
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]

            if 'train' in test_dir:
                labels = data[2].cuda()

            # print(labels.shape)

            ps = []
            for nn_idx in range(nn_num):
                # Generate explanations
                explanation = data[1]
                explanation = explanation[:, nn_idx:nn_idx+1, :, :, :]

                output, query, i2e_attn, e2i_attn = MODEL2(images=x, explanations=explanation, scores=model1_score)
                # breakpoint()
                p = torch.sigmoid(output)

                ps.append(p)

            # breakpoint()
            p = torch.stack(ps, dim=0).mean(dim=0)

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

            preds_val.append(preds)
            labels_val.append(labels)

            for j in range(x.shape[0]):
                pth = pths[j]

                if '_0_' in pth:
                    top1_cnt += 1

                    if results[j] == True:
                        top1_crt_cnt += 1

            running_corrects += torch.sum(preds == labels.data)

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