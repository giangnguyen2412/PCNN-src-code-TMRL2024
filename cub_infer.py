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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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

CATEGORY_ANALYSIS = True

if RunningParams.MODEL2_ADVISING is True:
    in_features = 2048
    print("Building FAISS index...! Training set is the knowledge base.")
    faiss_dataset = datasets.ImageFolder('/home/giang/Downloads/RN50_dataset_CUB_Pretraining/train',
                                         transform=Dataset.data_transforms['train'])

    faiss_data_loader = torch.utils.data.DataLoader(
        faiss_dataset,
        batch_size=RunningParams.batch_size,
        shuffle=False,  # turn shuffle to True
        num_workers=16,  # Set to 0 as suggested by
        # https://stackoverflow.com/questions/54773106/simple-way-to-load-specific-sample-using-pytorch-dataloader
        pin_memory=True,
    )

    HIGHPERFORMANCE_FEATURE_EXTRACTOR = True
    if HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
        INDEX_FILE = 'faiss/faiss_CUB200_class_idx_dict_HP_extractor.npy'
    else:
        INDEX_FILE = 'faiss/faiss_CUB200_class_idx_dict_LP_extractor.npy'
    if os.path.exists(INDEX_FILE):
        print("FAISS class index exists!")
        faiss_nns_class_dict = np.load(INDEX_FILE, allow_pickle="False", ).item()
        targets = faiss_data_loader.dataset.targets
        faiss_data_loader_ids_dict = dict()
        faiss_loader_dict = dict()
        for class_id in tqdm(range(len(faiss_data_loader.dataset.class_to_idx))):
            faiss_data_loader_ids_dict[class_id] = [x for x in range(len(targets)) if targets[x] == class_id] # check this value
            class_id_subset = torch.utils.data.Subset(faiss_dataset, faiss_data_loader_ids_dict[class_id])
            class_id_loader = torch.utils.data.DataLoader(class_id_subset, batch_size=128, shuffle=False)
            faiss_loader_dict[class_id] = class_id_loader

full_cub_dataset = ImageFolderForNNs('/home/giang/Downloads/datasets/CUB/combined',
                                         Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='best_model_tough-capybara-1414.pt',
                        help='Model check point')
    # parser.add_argument('--dataset', type=str,
    #                     default='balanced_val_dataset_6k',
    #                     help='Evaluation dataset')

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

    test_dir = '/home/giang/Downloads/RN50_dataset_CUB_LP/val'
    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForNNs(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    HIGHPERFORMANCE_MODEL1 = RunningParams.HIGHPERFORMANCE_MODEL1
    if HIGHPERFORMANCE_MODEL1 is True:
        from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

        resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load(
            'Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        resnet.load_state_dict(my_model_state_dict, strict=True)
        MODEL1 = resnet.cuda()
        MODEL1.eval()
        fc = list(MODEL1.children())[-1].cuda()
        fc = nn.DataParallel(fc)

        categorized_path = '/home/giang/Downloads/HP_INAT_RN50_dataset_CUB_backup'

    else:
        import torchvision
        inat_resnet = torchvision.models.resnet50(pretrained=True).cuda()
        inat_resnet.fc = nn.Sequential(nn.Linear(2048, 200)).cuda()
        my_model_state_dict = torch.load('50_vanilla_resnet_avg_pool_2048_to_200way.pth')
        inat_resnet.load_state_dict(my_model_state_dict, strict=True)
        MODEL1 = inat_resnet
        MODEL1.eval()

        fc = MODEL1.fc.cuda()
        fc = nn.DataParallel(fc)

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
            batch_size=50,
            shuffle=False,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )

        running_corrects = 0
        advising_crt_cnt = 0
        yes_cnt = 0
        true_cnt = 0
        uncertain_pred_cnt = 0

        categories = ['CorrectlyAccept', 'IncorrectlyAccept', 'CorrectlyReject', 'IncorrectlyReject']
        save_dir = '/home/giang/Downloads/advising_network/vis'
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
            # if batch_idx == 5:
            #     break
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
            embeddings = feature_extractor(x).flatten(start_dim=1)  # 512x1 for RN 18
            out = fc(embeddings)
            out = MODEL1(x)
            model1_p = torch.nn.functional.softmax(out, dim=1)
            model1_score, index = torch.topk(model1_p, 1, dim=1)
            _, model1_ranks = torch.topk(model1_p, 200, dim=1)
            predicted_ids = index.squeeze()

            # MODEL1 Y/N label for input x
            for sample_idx in range(x.shape[0]):
                query = pths[sample_idx]
                base_name = os.path.basename(query)

                if CATEGORY_ANALYSIS is True:
                    key = category_dict[base_name]
                    category_record[key]['total'] += 1

            model2_gt = (predicted_ids == gts) * 1  # 0 and 1

            labels = model2_gt
            # Get the idx of wrong predictions
            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]

            # Generate explanations
            if RunningParams.XAI_method == RunningParams.GradCAM:
                explanation = ModelExplainer.grad_cam(MODEL1, x, index, RunningParams.GradCAM_RNlayer, resize=False)
            elif RunningParams.XAI_method == RunningParams.NNs:
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
                    output, query, i2e_attn, e2i_attn = model(images=x, explanations=explanation, scores=model1_p)

                p = torch.nn.functional.softmax(output, dim=1)
                model2_score, preds = torch.max(p, 1)

                VISUALIZE_TRANSFORMER_ATTN = True
                if VISUALIZE_TRANSFORMER_ATTN is True:
                    i2e_attn = i2e_attn.mean(dim=1)
                    i2e_attn = i2e_attn[:, :, 1:]  # remove cls token

                    e2i_attn = e2i_attn.mean(dim=1)
                    e2i_attn = e2i_attn[:, :, 1:]  # remove cls token

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

                        bef_weights = i2e_attn[sample_idx][0:1]
                        aft_weights = e2i_attn[sample_idx][0:1]
                        query = pths[sample_idx]
                        base_name = os.path.basename(query)
                        # as the test images are from validation, then we do not need to exclude the fist prototype
                        # Visualize the attention b/w query vs the 1ST prototype
                        prototypes = data_loader.dataset.faiss_nn_dict[base_name][0:RunningParams.k_value]
                        Visualization.visualize_transformer_attn(bef_weights, aft_weights, prototypes[0], pths[sample_idx], title=model2_decision)

                REMOVE_UNCERT = False
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

                    gt_label = image_datasets['cub_test'].classes[gts[sample_idx].item()]
                    pred_label = image_datasets['cub_test'].classes[predicted_ids[sample_idx].item()]

                    model1_confidence = int(model1_score[sample_idx].item()*100)
                    model2_confidence = int(model2_score[sample_idx].item()*100)
                    model1_confidence_dist[model2_decision].append(model1_confidence)
                    model2_confidence_dist[model2_decision].append(model2_confidence)

                    # Finding the extreme cases
                    # if (action == 'Accept' and confidence < 50) or (action == 'Reject' and confidence > 80):
                    if True:
                        prototypes = data_loader.dataset.faiss_nn_dict[base_name][0:RunningParams.k_value]
                        # Visualization.visualize_model2_decision_with_prototypes(query,
                        #                                          gt_label,
                        #                                          pred_label,
                        #                                          model2_decision,
                        #                                          save_path,
                        #                                          save_dir,
                        #                                          model1_confidence,
                        #                                         model2_confidence,
                        #                                         prototypes)


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

        if RunningParams.M2_VISUALIZATION is True:
            for cat in categories:
                img_ratio = len(model1_confidence_dist[cat])*100/dataset_sizes[ds]
                title = '{}: {:.2f}% of test set'.format(cat, img_ratio)
            #     Visualization.visualize_histogram_from_list(data=model1_confidence_dist[cat],
            #                                                 title=title,
            #                                                 x_label='Confidence',
            #                                                 y_label='Images',
            #                                                 file_name=os.path.join(save_dir, cat + 'Confidence.pdf'),
            #                                                 )
            # #
            #     merge_pdf_cmd = 'img2pdf -o vis/{}Samples.pdf --rotation=ifvalid --pagesize A4^T vis/{}/*.jpg'.format(cat, cat)
            #     os.system(merge_pdf_cmd)
            #
            # compress_cmd = 'tar -czvf analysis.tar.gz vis/*.pdf'
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

        print(category_record)
        if CATEGORY_ANALYSIS is True:
            for c in correctness_bins:
                print("{} - {:.2f}".format(c, category_record[c]['crt']*100/category_record[c]['total']))