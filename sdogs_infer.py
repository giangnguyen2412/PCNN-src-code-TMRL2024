import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torchvision import datasets, models, transforms

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, StanfordDogsDataset, ImageFolderForNNs
from helpers import HelperFunctions
from explainers import ModelExplainer
from transformer import Transformer_AdvisingNetwork
from visualize import Visualization
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as T

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
ModelExplainer = ModelExplainer()
Visualization = Visualization()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

CATEGORY_ANALYSIS = False

import math
def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return T.Compose([
        T.Resize((height, width)),
        T.Pad(pad_values),
        T.ToTensor(),
        T.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)
full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/advising_network/stanford-dogs/data/images/BACKUP/train',
                                     preprocess)

# Because it has a unique type of mapping the labels, we need reference_dataset to convert the predicted ids (StanfordDogsDataset) to ImageFolder ids
reference_dataset = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_model_eager-cloud-3132.pt',
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
    test_dir = '/home/giang/Downloads/advising_network/stanford-dogs/data/images/test'  ##################################
    # test_dir = '/home/giang/Downloads/advising_network/stanford-dogs/data/images/validation_top2'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, preprocess)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    model = models.resnet34(pretrained=True)

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 120)

    from collections import OrderedDict

    new_ckpt = OrderedDict()
    ckpt = torch.load('/home/giang/Downloads/advising_network/stanford-dogs/HAL9002_RN34.pt')
    print(ckpt['val_acc'] / 100)

    for k, v in ckpt['model_state_dict'].items():
        new_k = k.replace('module.', '')
        new_ckpt[new_k] = v

    model.load_state_dict(new_ckpt)

    MODEL1 = model.cuda()
    MODEL1.eval()

    torch.manual_seed(42)

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=8,
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
        save_dir = '/home/giang/Downloads/advising_network/vis'
        if RunningParams.M2_VISUALIZATION is True:
            HelperFunctions.check_and_rm(save_dir)
            HelperFunctions.check_and_mkdir(save_dir)

        # save_dir = '/home/giang/Downloads/advising_network/attn_maps'
        # if RunningParams.M2_VISUALIZATION is True:
        #     HelperFunctions.check_and_rm(save_dir)
        #     HelperFunctions.check_and_mkdir(save_dir)

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
            if batch_idx == 40:
                break
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
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            model1_score, index = torch.topk(model1_p, 1, dim=1)
            import pdb

            pdb.set_trace = lambda: 1
            pdb.set_trace()
            predicted_ids = index.squeeze()
            # MODEL1 Y/N label for input x
            for sample_idx in range(x.shape[0]):
                query = pths[sample_idx]
                base_name = os.path.basename(query)

            # MODEL1 Y/N label for input x
            ########################################################################
            for sample_idx in range(x.shape[0]):
                predicted_idx = predicted_ids[sample_idx]
                dog_name = reference_dataset.mapping[predicted_idx.item()]
                predicted_ids[sample_idx] = data_loader.dataset.class_to_idx[dog_name]
            ########################################################################

            model2_gt = (predicted_ids == gts) * 1  # 0 and 1
            labels = model2_gt

            # For validation, we need to get it from the dict because we manipulate its MODEL2 labels (e.g. same input but NNS from different species)
            if 'train' in test_dir or 'val' in test_dir:
                labels = data[2].cuda()

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
                    model1_score.fill_(1 / 120)

                output, query, i2e_attn, e2i_attn = MODEL2(images=x, explanations=explanation, scores=model1_score)

                # convert logits to probabilities using sigmoid function
                p = torch.sigmoid(output)

                pdb.set_trace()

                # classify inputs as 0 or 1 based on the threshold of 0.5
                preds = (p >= 0.5).long().squeeze()
                model2_score = p

                results = (preds == labels)

                ################################
                preds_val.append(preds)
                labels_val.append(labels)
                ################################

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

            VISUALIZE_TRANSFORMER_ATTN = True
            if VISUALIZE_TRANSFORMER_ATTN is True:
                pdb.set_trace()

                i2e_attn = i2e_attn.mean(dim=1)  # bsx8x3x50
                i2e_attn = i2e_attn[:, :, 1:]  # remove cls token --> bsx3x49

                e2i_attn = e2i_attn.mean(dim=1)
                e2i_attn = e2i_attn[:, :, 1:]  # remove cls token

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
                    base_name = os.path.basename(query)
                    prototypes = data_loader.dataset.faiss_nn_dict[base_name][0:RunningParams.k_value]
                    for prototype_idx in range(RunningParams.k_value):
                        bef_weights = i2e_attn[sample_idx, prototype_idx:prototype_idx + 1, :]
                        aft_weights = e2i_attn[sample_idx, prototype_idx:prototype_idx + 1, :]

                        # as the test images are from validation, then we do not need to exclude the fist prototype
                        prototype = prototypes[prototype_idx]
                        Visualization.visualize_transformer_attn_sdogs(bef_weights, aft_weights, prototype, query,
                                                                    model2_decision, prototype_idx)

                    # Combine visualizations
                    # cmd = 'montage tmp/{}/{}_[0-{}].png -tile 3x1 -geometry +0+0 tmp/{}/{}.jpeg'.format(
                    #     model2_decision, base_name, RunningParams.k_value - 1, model2_decision, base_name)

                    # os.system(cmd)
                    # Remove unused images
                    # cmd = 'rm -rf tmp/{}/{}_[0-{}].png'.format(
                    #     model2_decision, base_name, RunningParams.k_value - 1)
                    # os.system(cmd)

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

        ################################################################

        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/attn_maps/IncorrectlyAccept/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/attn_maps/IncorrectlyAccept/*.jpg'
        os.system(cmd)
        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/attn_maps/IncorrectlyReject/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/attn_maps/IncorrectlyReject/*.jpg'
        os.system(cmd)

        ################################################################
        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/vis/IncorrectlyAccept/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/vis/IncorrectlyAccept/*.jpg'
        os.system(cmd)
        cmd = 'img2pdf -o /home/giang/Downloads/advising_network/vis/IncorrectlyReject/output.pdf ' \
              '--pagesize A4^T /home/giang/Downloads/advising_network/vis/IncorrectlyReject/*.jpg'
        os.system(cmd)
        #################################################################

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

        if RunningParams.MODEL2_ADVISING is True:
            advising_acc = advising_crt_cnt.double() / len(image_datasets[ds])

            print(
                '{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. Acc: {:.2f} - Correction Acc: {:.2f}'.format(
                    os.path.basename(test_dir), epoch_acc * 100, yes_ratio * 100, true_ratio * 100, advising_acc * 100))
            print('Original misclassified: {} - After correction: {}'.format(orig_wrong, adv_wrong))
        else:
            print('{} - Binary Acc: {:.2f} - MODEL2 Yes Ratio: {:.2f} - Orig. accuracy: {:.2f}'.format(
                os.path.basename(test_dir), epoch_acc * 100, yes_ratio * 100, true_ratio * 100))

        np.save('confidence.npy', confidence_dict)
        print('top1 = {}'.format(top1_crt_cnt/1200))