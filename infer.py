import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import faiss
import matplotlib.pyplot as plt

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


RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()
ModelExplainer = ModelExplainer()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"


CATEGORY_ANALYSIS = True
if CATEGORY_ANALYSIS is True:
    import glob
    correctness = ['Correct', 'Wrong']
    diffs = ['Easy', 'Medium', 'Hard']
    category_dict = {}
    category_record = {}

    for c in correctness:
        for d in diffs:
            dir = os.path.join('/home/giang/Downloads/RN18_dataset_val', c, d)
            files = glob.glob(os.path.join(dir, '*', '*.*'))
            key = c + d
            for file in files:
                base_name = os.path.basename(file)
                category_dict[base_name] = key

            category_record[key] = {}
            category_record[key]['total'] = 0
            category_record[key]['crt'] = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_model_eternal-armadillo-847.pt',
                        help='Model check point')
    parser.add_argument('--dataset', type=str,
                        default='balanced_val_dataset_6k',
                        help='Evaluation dataset')

    args = parser.parse_args()
    model_path = os.path.join('best_models', args.ckpt)
    print(args)

    model = Transformer_AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    print('Validation accuracy: {:.2f}'.format(acc))
    print(RunningParams.__dict__)

    model.eval()

    data_dir = '/home/giang/Downloads/datasets/'
    # TODO: Using mask to make the predictions compatible with ImageNet-R, ObjectNet, ImageNet-A
    val_datasets = Dataset.test_datasets
    # val_datasets = [args.dataset]
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), Dataset.data_transforms['val'])
    #                   for x in val_datasets}
    if RunningParams.XAI_method == RunningParams.NNs:
        image_datasets = {x: ImageFolderForNNs(os.path.join(data_dir, x), Dataset.data_transforms['val'])
                          for x in val_datasets}
    else:
        image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), Dataset.data_transforms['val'])
                          for x in val_datasets}

    dataset_sizes = {x: len(image_datasets[x]) for x in val_datasets}

    model1_name = 'resnet18'
    MODEL1 = models.resnet18(pretrained=True).eval()
    fc = MODEL1.fc
    fc = fc.cuda()

    feature_extractor = nn.Sequential(*list(MODEL1.children())[:-1])  # avgpool feature
    feature_extractor.cuda()
    feature_extractor = nn.DataParallel(feature_extractor)

    for ds in val_datasets:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=RunningParams.batch_size,
            shuffle=False,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
        )

        running_corrects = 0
        advising_crt_cnt = 0
        yes_cnt = 0
        true_cnt = 0
        uncertain_pred_cnt = 0

        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']
        save_dir = '/home/giang/Downloads/advising_network/vis/'
        HelperFunctions.check_and_rm(save_dir)
        HelperFunctions.check_and_mkdir(save_dir)
        confidence_dist = dict()
        for cat in categories:
            confidence_dist[cat] = list()
            HelperFunctions.check_and_mkdir(os.path.join(save_dir, cat))

        infer_result_dict = dict()

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            if RunningParams.XAI_method == RunningParams.NNs:
                x = data[0].cuda()
            else:
                x = data.cuda()
            gts = gt.cuda()

            # Step 1: Forward pass input x through MODEL1 - Explainer
            embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18
            out = fc(embeddings)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            score, index = torch.topk(model1_p, 1, dim=1)
            _, model1_ranks = torch.topk(model1_p, 1000, dim=1)
            predicted_ids = index.squeeze()

            # MODEL1 Y/N label for input x
            if RunningParams.IMAGENET_REAL and ds == Dataset.IMAGENET_1K:
                model2_gt = torch.zeros([x.shape[0]], dtype=torch.int64).cuda()
                for sample_idx in range(x.shape[0]):
                    query = pths[sample_idx]
                    base_name = os.path.basename(query)

                    if CATEGORY_ANALYSIS is True:
                        key = category_dict[base_name]
                        category_record[key]['total'] += 1


                    real_ids = Dataset.real_labels[base_name]
                    # TODO: Ignore images having no labels
                    if predicted_ids[sample_idx].item() in real_ids:
                        model2_gt[sample_idx] = 1
                    else:
                        model2_gt[sample_idx] = 0
            else:
                model2_gt = (predicted_ids == gts) * 1  # 0 and 1

            labels = model2_gt
            # Get the idx of wrong predictions
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]

            # Generate explanations
            if RunningParams.XAI_method == RunningParams.GradCAM:
                explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
            elif RunningParams.XAI_method == RunningParams.NNs:
                embeddings = embeddings.cpu().detach().numpy()
                if RunningParams.PRECOMPUTED_NN is True:
                    explanation = data[1]
                    explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                    # Make the random epplanations
                    # Pseudo-code --> Need to debug here to see
                    # random_exp = torch.random(explanation[0].shape)
                    # explanation[idx_0] = random_exp
                else:
                    print("Error: Not implemented yet!")
                    exit(-1)

            if RunningParams.advising_network is True:
                # Forward input, explanations, and softmax scores through MODEL2
                if RunningParams.XAI_method == RunningParams.NO_XAI:
                    output, _, _ = model(images=x, explanations=None, scores=model1_p)
                else:
                    output, query, nns, _ = model(images=x, explanations=explanation, scores=model1_p)

                p = torch.nn.functional.softmax(output, dim=1)
                _, preds = torch.max(p, 1)

                REMOVE_UNCERT = True
                for sample_idx in range(x.shape[0]):
                    pred = preds[sample_idx].item()
                    model2_conf = p[sample_idx][pred].item()
                    if model2_conf < 0.60:
                        uncertain_pred_cnt += 1
                        if REMOVE_UNCERT is True:
                            preds[sample_idx] = -1

                if CATEGORY_ANALYSIS is True:
                    for sample_idx in range(x.shape[0]):
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)

                        key = category_dict[base_name]
                        if preds[sample_idx].item() == labels[sample_idx].item():
                            category_record[key]['crt'] += 1

            # Running ADVISING process
            # TODO: Reduce compute overhead by only running on disagreed samples
            advising_steps = RunningParams.advising_steps
            # When using advising & still disagree & examining only top $advising_steps + 1
            while RunningParams.MODEL2_ADVISING is True and sum(preds) > 0:
                for k in range(1, RunningParams.advising_steps):
                    # top-k predicted labels
                    model1_topk = model1_ranks[:, k]
                    for pred_idx in range(x.shape[0]):
                        if preds[pred_idx] == 0:
                            # Change the predicted id to top-k if MODEL2 disagrees
                            predicted_ids[pred_idx] = model1_topk[pred_idx]
                    # Unsqueeze to fit into grad_cam()
                    index = torch.unsqueeze(predicted_ids, dim=1)
                    # Generate heatmap explanation using new category ids
                    advising_saliency = grad_cam(MODEL1, x, index, saliency_layer=RunningParams.GradCAM_RNlayer, resize=True)

                    # TODO: If the explanation is No-XAI, the inputs in advising process don't change
                    if RunningParams.advising_network is True:
                        advising_output = model(x, advising_saliency, model1_p)
                        advising_p = torch.nn.functional.softmax(advising_output, dim=1)
                        _, preds = torch.max(advising_p, 1)
                break

            # If MODEL2 still disagrees, we revert the ids to top-1 predicted labels
            model1_top1 = model1_ranks[:, 0]
            for pred_idx in range(x.shape[0]):
                if preds[pred_idx] == 0:
                    # Change the predicted id to top-k if MODEL2 disagrees
                    predicted_ids[pred_idx] = model1_top1[pred_idx]

            # statistics
            if RunningParams.MODEL2_ADVISING:
                # Re-compute the accuracy of imagenet classification
                advising_crt_cnt += torch.sum(predicted_ids == gts)
                advising_labels = (predicted_ids == gts) * 1
                running_corrects += torch.sum(preds == advising_labels.data)
            else:
                running_corrects += torch.sum(preds == labels.data)

            if RunningParams.M2_VISUALIZATION is True:
                results = (preds == labels)
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

                    gt_label = HelperFunctions.label_map.get(gts[sample_idx].item()).split(",")[0]
                    gt_label = gt_label[0].lower() + gt_label[1:]

                    pred_label = HelperFunctions.label_map.get(predicted_ids[sample_idx].item()).split(",")[0]
                    pred_label = pred_label[0].lower() + pred_label[1:]

                    confidence = int(score[sample_idx].item()*100)
                    confidence_dist[model2_decision].append(confidence)

                    if RunningParams.IMAGENET_REAL is True and ds == Dataset.IMAGENET_1K:
                        real_ids = Dataset.real_labels[base_name]
                        gt_labels = []
                        for id in real_ids:
                            gt_label = HelperFunctions.label_map.get(id).split(",")[0]
                            gt_label = gt_label[0].lower() + gt_label[1:]
                            gt_labels.append(gt_label)

                        gt_label = '|'.join(gt_labels)
                    # Finding the extreme cases
                    if (action == 'Accept' and confidence < 50) or (action == 'Reject' and confidence > 80):
                        # Visualization.visualize_model2_decisions(query,
                        #                                          gt_label,
                        #                                          pred_label,
                        #                                          model2_decision,
                        #                                          save_path,
                        #                                          save_dir,
                        #                                          confidence)
                        pass

                    infer_result_dict[base_name] = dict()
                    infer_result_dict[base_name]['model2_decision'] = model2_decision
                    infer_result_dict[base_name]['confidence'] = confidence
                    infer_result_dict[base_name]['gt_label'] = gt_label
                    infer_result_dict[base_name]['pred_label'] = pred_label
                    infer_result_dict[base_name]['result'] = result is True

            yes_cnt += sum(preds)
            true_cnt += sum(labels)
        np.save('infer_results/{}.npy'.format(args.ckpt), infer_result_dict)

        if RunningParams.M2_VISUALIZATION is True:
            for cat in categories:
                img_ratio = len(confidence_dist[cat])*100/dataset_sizes[ds]
                title = '{}: {:.2f}'.format(cat, img_ratio)
                # Visualization.visualize_histogram_from_list(data=confidence_dist[cat],
                #                                             title=title,
                #                                             x_label='Confidence',
                #                                             y_label='Images',
                #                                             file_name=os.path.join(save_dir, cat + '.pdf'),
                #                                             )

                # merge_pdf_cmd = 'img2pdf -o vis/{}Samples.pdf --rotation=ifvalid --pagesize A4^T vis/{}/*.JPEG'.format(cat, cat)
                # os.system(merge_pdf_cmd)

            # compress_cmd = 'tar -czvf analysis.tar.gz vis/'
            # os.system(compress_cmd)

        epoch_acc = running_corrects.double() / len(image_datasets[ds])
        yes_ratio = yes_cnt.double() / len(image_datasets[ds])
        true_ratio = true_cnt.double() / len(image_datasets[ds])
        uncertain_ratio = uncertain_pred_cnt / len(image_datasets[ds])
        if REMOVE_UNCERT is True:
            epoch_acc = running_corrects.double() / (len(image_datasets[ds]) - uncertain_pred_cnt)

        if RunningParams.MODEL2_ADVISING is True:
            advising_acc = advising_crt_cnt.double() / len(image_datasets[ds])

            print('{} - Acc: {:.2f} - Yes Ratio: {:.2f} - Orig. accuracy: {:.2f} - Advising. accuracy: {:.2f}'.format(
                ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100, advising_acc * 100))
        else:
            if REMOVE_UNCERT is True:
                print('{} - Acc: {:.2f} - Always say Yes: {:.2f} - Uncertain. ratio: {:.2f}'.format(
                    ds, epoch_acc * 100, true_ratio * 100, 0 * 100))
            else:
                print('{} - Acc: {:.2f} - Yes Ratio: {:.2f} - Always say Yes: {:.2f} - Uncertain. ratio: {:.2f}'.format(
                    ds, epoch_acc * 100, yes_ratio * 100, true_ratio * 100, uncertain_ratio * 100))

        if CATEGORY_ANALYSIS is True:
            for c in correctness:
                for d in diffs:
                    print("{} - {} - {:.2f}".format(c, d, category_record[c+d]['crt']*100/category_record[c+d]['total']))
