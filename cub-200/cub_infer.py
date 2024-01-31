# 1. Infer the binary performance of AdvNet on the CUB-200 test set
# 2. Visualize the attention maps of the transformer
# 3. Compute human-AI team accuracy
# 4. Visualize AdvNet binary decisions
import torch
import torch.nn as nn
import numpy as np
import os
import argparse

import sys
sys.path.insert(0, '/home/giang/Downloads/advising_network')


from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForNNs
from helpers import HelperFunctions
from transformer import Transformer_AdvisingNetwork, CNN_AdvisingNetwork, ViT_AdvisingNetwork
from visualize import Visualization
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
Visualization = Visualization()


if RunningParams.set != 'test':
    print('Please setting the set to test in params.py! Exiting...')
    exit(-1)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

CATEGORY_ANALYSIS = False

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_' + RunningParams.wandb_sess_name + '.pt',
                        # default='best_model_' + 'bs256-p1.0-dropout-0.0-NNth-1' + '.pt',
                        # default='best_model_different-grass-3212.pt', # Normal model
                        # default='best_model_blooming-sponge-3236.pt', # 2nd NNs
                        default='best_model_decent-pyramid-3156.pt', # Normal model, top10, run1
                        # default='best_model_avid-cosmos-3201.pt', # M=N=L=1 model
                        # default='best_model_serene-sound-3240.pt',  # M=N=L=2 model
                        # default='best_model_rare-shadow-3213.pt', # M=N=L=1 model convs only
                        # default='best_model_decent-mountain-3215.pt', # M=N=L=1 model convs only, no SA
                        # default='best_model_legendary-durian-3216.pt', # Random negative samples data sampling, RN50, 1st NN

                        # default='best_model_lilac-waterfall-3238.pt',  # 3rd NNs

                        # default='best_model_faithful-rain-3211.pt', # M=N=L=1 model convs only, no SA
                        # default='best_model_cosmic-waterfall-3174.pt', # no CA --> have no cross attn to visualize
                        # default='best_model_young-planet-3170.pt',  # no SA --> clear heatmaps
                        # default='best_model_serene-field-3176.pt',  # no SA --> depth2 head2
                        # default='best_model_warm-disco-3178.pt',  # no SA --> depth2 head1
                        # default='best_model_cerulean-sponge-3186.pt',  # Normal model, top15
                        # default='best_model_eager-field-3187.pt',  # Normal model, top10, run2
                        # default='best_model_light-cosmos-3188.pt',  # Normal model, top10, run3
                        # default='best_model_colorful-dragon-3182.pt',  # Normal model, top5
                        # default='best_model_prime-forest-3183.pt',  # Normal model, top3
                        # default='best_model_skilled-night-3167.pt',  # no augmentation
                        # default='best_model_olive-dream-3169.pt',  # no training conv layers
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)
    print(args)

    if RunningParams.VisionTransformer is True:
        MODEL2 = ViT_AdvisingNetwork()
    else:
        if RunningParams.TRANSFORMER_ARCH == True:
            MODEL2 = Transformer_AdvisingNetwork()
        else:
            MODEL2 = CNN_AdvisingNetwork()

    MODEL2 = nn.DataParallel(MODEL2).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    MODEL2.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    print('Model accuracy: {:.2f}'.format(acc))

    print(RunningParams.__dict__)

    MODEL2.eval()
    test_dir = f'{RunningParams.parent_dir}/datasets/CUB/test0'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    if RunningParams.VisionTransformer is True:
        import timm

        class CustomViT(nn.Module):
            def __init__(self, base_model):
                super(CustomViT, self).__init__()
                self.base_model = base_model

            def forward(self, x):
                # Get the features from the base ViT model
                x = self.base_model.forward_features(x)
                # Extract the CLS token (first token)
                cls_token = x[:, 0]
                # Pass the features through the classifier
                output = self.base_model.head(cls_token)
                return output, cls_token


        # Initialize the base model and load the trained weights
        base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=200)
        model_path = "./vit_base_patch16_224_cub_200way.pth"
        state_dict = torch.load(model_path, map_location=torch.device("cuda"))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(new_state_dict)

        # Wrap the base model in the custom model
        model = CustomViT(base_model)
        MODEL1 = model.cuda()
    else:

        from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

        resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/iNaturalist_pretrained_RN50_85.83.pth')

        # TODO: Define INAT and IMAGENET features to be used in two cases of using RN50

        if RunningParams.resnet == 50 and RunningParams.RN50_INAT is False:
            resnet = models.resnet50(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/cub-200/imagenet_pretrained_resnet50_cub_200way_top1acc_63.pth')
        elif RunningParams.resnet == 34:
            resnet = models.resnet34(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/cub-200/imagenet_pretrained_resnet34_cub_200way_top1acc_62_81.pth')
        elif RunningParams.resnet == 18:
            resnet = models.resnet18(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(512, 200)).cuda()
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/cub-200/imagenet_pretrained_resnet18_cub_200way_top1acc_60_22.pth')

        resnet.load_state_dict(my_model_state_dict, strict=True)
        if RunningParams.resnet == 34 or RunningParams.resnet == 18 or (
                RunningParams.resnet == 50 and RunningParams.RN50_INAT is False):
            resnet.fc = resnet.fc[0]

        MODEL1 = resnet.cuda()

    MODEL1.eval()

    categorized_path = f'{RunningParams.parent_dir}/RN50_dataset_CUB_HIGH/combined'

    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=17,
            shuffle=True,  # turn shuffle to False
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
        advnet_confidence_dict = dict()
        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']

        for cat in categories:
            advnet_confidence_dict[cat] = list()

        save_dir = f'{RunningParams.prj_dir}/vis/cub/'
        if RunningParams.M2_VISUALIZATION is True:
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

        SAME = False
        RAND = False
        RAND_REAL = False

        print('Sanity checks in binary classification!')
        print(SAME, RAND, RAND_REAL)

        if True:
            accpt_cnt = 0
            accpt_conf = 0

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
            # Step 1: Forward pass input x through MODEL1 - Explainer
            if RunningParams.VisionTransformer is True:
                out, _ = MODEL1(x)
            else:
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

            # Generate explanations
            if RunningParams.XAI_method == RunningParams.NNs:
                explanation = data[1]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]
                # Find the maximum value along each row
                max_values, _ = torch.max(model1_p, dim=1)

                if SAME is True:
                    model1_score.fill_(0.999)

                    explanation = x.clone().unsqueeze(1).repeat(1, RunningParams.k_value, 1, 1, 1)
                if RAND is True:
                    torch.manual_seed(42)
                    explanation = torch.rand_like(explanation)
                    explanation.cuda()
                    # Replace the maximum value with random guess
                    model1_score.fill_(1 / 200)

                if RAND_REAL is True:
                    torch.manual_seed(42)
                    explanation = explanation[torch.randperm(explanation.size(0))]

                # Forward input, explanations, and softmax scores through MODEL2
                output, query, i2e_attn, e2i_attn = MODEL2(images=x, explanations=explanation, scores=model1_score)

                # convert logits to probabilities using sigmoid function
                p = torch.sigmoid(output)

                # classify inputs as 0 or 1 based on the threshold of 0.5
                preds = (p >= 0.5).long().squeeze()

                if True:
                    accpt_conf += (torch.sigmoid(output.squeeze()) * preds).sum().item()
                    accpt_cnt += sum(preds).item()

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

                optimal_T = 95
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

                VISUALIZE_TRANSFORMER_ATTN = RunningParams.VISUALIZE_TRANSFORMER_ATTN
                if VISUALIZE_TRANSFORMER_ATTN is True:
                    # i2e_attn = i2e_attn.mean(dim=1)  # bsx8x3x50
                    # i2e_attn = i2e_attn[:, :, 1:]  # remove cls token --> bsx3x49
                    #
                    # e2i_attn = e2i_attn.mean(dim=1)
                    # e2i_attn = e2i_attn[:, :, 1:]  # remove cls token

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

                        if pred == 0:
                            conf_score = 1 - model2_score[sample_idx].item()
                        else:
                            conf_score = model2_score[sample_idx].item()

                        advnet_confidence_dict[model2_decision].append(conf_score)

                        # breakpoint()
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)
                        # breakpoint()
                        # prototypes = data_loader.dataset.faiss_nn_dict[base_name]['NNs'][0:RunningParams.k_value]
                        # for prototype_idx in range(RunningParams.k_value):
                        #     bef_weights = i2e_attn[sample_idx, prototype_idx:prototype_idx + 1, :]
                        #     aft_weights = e2i_attn[sample_idx, prototype_idx:prototype_idx + 1, :]
                        #
                        #     # as the test images are from validation, then we do not need to exclude the fist prototype
                        #     prototype = prototypes[prototype_idx]
                            # Visualization.visualize_transformer_attn_birds(bef_weights, aft_weights, prototype, query,
                            #                                             model2_decision, prototype_idx)

                        # Combine visualizations
                        # cmd = 'montage attn_maps/cub/{}/{}_[0-{}].png -tile 3x1 -geometry +0+0 tmp/{}/{}.jpeg'.format(
                        #     model2_decision, base_name, RunningParams.k_value - 1, model2_decision, base_name)
                        # cmd = 'montage tmp/{}/{}_[0-{}].png -tile 3x1 -geometry +0+0 my_plot.png'.format(
                        #     model2_decision, base_name, RunningParams.k_value - 1)
                        # os.system(cmd)
                        # Remove unused images
                        # cmd = 'rm -rf attn_maps/cub/{}/{}_[0-{}].png'.format(
                        #     model2_decision, base_name, RunningParams.k_value - 1)
                        # os.system(cmd)

                    # cmd = f'img2pdf -o {RunningParams.prj_dir}/attn_maps/cub/IncorrectlyAccept/output.pdf ' \
                    #       f'--pagesize A4^T {RunningParams.prj_dir}/attn_maps/cub/IncorrectlyAccept/*.png'
                    # os.system(cmd)
                    #
                    # cmd = f'img2pdf -o {RunningParams.prj_dir}/attn_maps/cub/CorrectlyAccept/output.pdf ' \
                    #       f'--pagesize A4^T {RunningParams.prj_dir}/attn_maps/cub/CorrectlyAccept/*.png'
                    # os.system(cmd)
                    #
                    # cmd = f'img2pdf -o {RunningParams.prj_dir}/attn_maps/cub/IncorrectlyReject/output.pdf ' \
                    #       f'--pagesize A4^T {RunningParams.prj_dir}/attn_maps/cub/IncorrectlyReject/*.png'
                    # os.system(cmd)
                    #
                    # cmd = f'img2pdf -o {RunningParams.prj_dir}/attn_maps/cub/CorrectlyReject/output.pdf ' \
                    #       f'--pagesize A4^T {RunningParams.prj_dir}/attn_maps/cub/CorrectlyReject/*.png'
                    # os.system(cmd)

                AI_DELEGATE = True
                if AI_DELEGATE is True:
                    for sample_idx in range(x.shape[0]):
                        result = results[sample_idx].item()
                        pred = preds[sample_idx].item()

                        s = int(model1_score[sample_idx].item() * 100)
                        if s not in confidence_dict:
                            # First: number of predictions having this confidence
                            # Second: number of predictions having this confidence
                            # Third: number of correct samples having this confidence
                            # The confidence is from MODEL1 output
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

                    # model1_confidence = int(model1_score[sample_idx].item() * 100)
                    # model2_confidence = int(model2_score[sample_idx].item() * 100)

                    model1_confidence = model1_score[sample_idx].item()
                    model2_confidence = model2_score[sample_idx].item()

                    model1_confidence_dist[model2_decision].append(model1_confidence)
                    model2_confidence_dist[model2_decision].append(model2_confidence)
                    # breakpoint()

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

        cmd = f'img2pdf -o {RunningParams.prj_dir}/vis/IncorrectlyAccept/output.pdf ' \
              f'--pagesize A4^T {RunningParams.prj_dir}/vis/IncorrectlyAccept/*.jpg'
        os.system(cmd)
        cmd = f'img2pdf -o {RunningParams.prj_dir}/vis/IncorrectlyReject/output.pdf ' \
              f'--pagesize A4^T {RunningParams.prj_dir}/vis/IncorrectlyReject/*.jpg'
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

        print(f'Accpt conf: {accpt_conf/accpt_cnt}')

        # for k, v in advnet_confidence_dict.items():
        #     advnet_confidence_dict[k] = sum(v) / len(v)

        # print(f'advnet_confidence_dict: {advnet_confidence_dict}')
        np.save('confidence.npy', confidence_dict)