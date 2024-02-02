import sys
sys.path.append('/home/giang/Downloads/advising_network')

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

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
Visualization = Visualization()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

CATEGORY_ANALYSIS = False

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/Stanford_Dogs_dataset/train',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_' + RunningParams.wandb_sess_name + '.pt',
                        # default='best_model_copper-moon-3322.pt',  # RN18
                        default='best_model_woven-deluge-3324.pt',  # RN34
                        # default='best_model_dainty-blaze-3325.pt',  # RN50 run 1
                        # default='best_model_quiet-bee-3327.pt',  # RN50 run 2
                        # default='best_model_likely-dragon-3328.pt',  # RN50 run 3
                        # default='best_model_driven-smoke-3329.pt',  # RN50 no augmentation
                        help='Model check point')

    args = parser.parse_args()
    # model_path = os.path.join('best_models', args.ckpt)
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)

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

    print(f'e{epoch} - loss: {loss} - Model acc: {acc}')

    print(RunningParams.__dict__)

    MODEL2.eval()
    test_dir = f'{RunningParams.parent_dir}/Stanford_Dogs_dataset/test'

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    ################################################################

    import torchvision

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if RunningParams.resnet == 50:
        model = torchvision.models.resnet50(pretrained=True).cuda()
        model.fc = nn.Linear(2048, 120)
    elif RunningParams.resnet == 34:
        model = torchvision.models.resnet34(pretrained=True).cuda()
        model.fc = nn.Linear(512, 120)
    elif RunningParams.resnet == 18:
        model = torchvision.models.resnet18(pretrained=True).cuda()
        model.fc = nn.Linear(512, 120)

    print('{}/stanford-dogs/resnet{}_stanford_dogs.pth'.format(RunningParams.prj_dir, RunningParams.resnet))
    my_model_state_dict = torch.load(
        '{}/stanford-dogs/resnet{}_stanford_dogs.pth'.format(RunningParams.prj_dir, RunningParams.resnet),
        map_location='cuda'
    )
    new_state_dict = {k.replace("model.", ""): v for k, v in my_model_state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)

    MODEL1 = model.cuda()
    MODEL1.eval()

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                         ])

    ################################################################

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

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()
        advnet_confidence_dict = dict()
        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']

        for cat in categories:
            advnet_confidence_dict[cat] = list()

        save_dir = f'{RunningParams.prj_dir}/vis/dogs/'
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

            if len(data_loader.dataset.classes) < 120:
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
                    model1_score.fill_(1 / 120)

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

                VISUALIZE_COMPARATOR_HEATMAPS = RunningParams.VISUALIZE_COMPARATOR_HEATMAPS
                if VISUALIZE_COMPARATOR_HEATMAPS is True:
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
            # np.save('infer_results/{}.npy'.format(args.ckpt), infer_result_dict)

        print(running_corrects, len(image_datasets[ds]))
        epoch_acc = running_corrects.double() / len(image_datasets[ds])
        yes_ratio = yes_cnt.double() / len(image_datasets[ds])
        true_ratio = true_cnt.double() / len(image_datasets[ds])

        orig_wrong = len(image_datasets[ds]) - true_cnt
        adv_wrong = len(image_datasets[ds]) - advising_crt_cnt

        print('{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. accuracy: {:.2f}'.format(
            os.path.basename(test_dir), epoch_acc * 100, yes_ratio * 100, true_ratio * 100))

        print(f'Accpt conf: {accpt_conf/accpt_cnt}')

        # for k, v in advnet_confidence_dict.items():
        #     advnet_confidence_dict[k] = sum(v) / len(v)

        # print(f'advnet_confidence_dict: {advnet_confidence_dict}')
        np.save('confidence.npy', confidence_dict)