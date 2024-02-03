# Run advising process using AdvNet
import torch
import torch.nn as nn
import os
import argparse

import sys
sys.path.insert(0, '/home/giang/Downloads/advising_network')


from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams('DOGS')

Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

torch.manual_seed(42)

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/{RunningParams.test_path}',
                                     Dataset.data_transforms['train'])

PRODUCT_OF_EXPERTS = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_copper-moon-3322.pt',  # RN18
                        # default='best_model_woven-deluge-3324.pt',  # RN34
                        # default='best_model_dainty-blaze-3325.pt',  # RN50 run 1
                        # default='best_model_quiet-bee-3327.pt',  # RN50 run 2
                        default='best_model_likely-dragon-3328.pt',  # RN50 run 3
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)

    if PRODUCT_OF_EXPERTS is True:
        ################################################################################
        import torchvision

        if RunningParams.resnet == 50:
            model = torchvision.models.resnet50(pretrained=True).cuda()
            model.fc = nn.Linear(2048, 120)
        elif RunningParams.resnet == 34:
            model = torchvision.models.resnet34(pretrained=True).cuda()
            model.fc = nn.Linear(512, 120)
        elif RunningParams.resnet == 18:
            model = torchvision.models.resnet18(pretrained=True).cuda()
            model.fc = nn.Linear(512, 120)

        print(f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth')
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/dogs-120/resnet{RunningParams.resnet}_stanford_dogs.pth',
            map_location='cuda'
        )
        new_state_dict = {k.replace("model.", ""): v for k, v in my_model_state_dict.items()}

        model.load_state_dict(new_state_dict, strict=True)

        MODEL1 = model
        MODEL1 = nn.DataParallel(MODEL1).cuda()
        MODEL1.eval()
        correct_cnt_model1 = 0

        ###############################################################################

    print(args)

    MODEL2 = Transformer_AdvisingNetwork()
    model = nn.DataParallel(MODEL2).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']
    RunningParams.XAI_method = running_params.XAI_method

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    f1 = checkpoint['best_f1']
    print(epoch)

    print('Validation accuracy: {:.4f}'.format(acc))
    print('F1 score: {:.4f}'.format(f1))

    print(f'PRODUCT_OF_EXPERTS: {PRODUCT_OF_EXPERTS}')

    model.eval()

    # test_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/test'  ##################################
    test_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=17,
            shuffle=False,  # turn shuffle to False
            num_workers=16,
            pin_memory=True,
            drop_last=False  # Do not remove drop last because it affects performance
        )

        running_corrects = 0
        running_corrects_top5 = 0
        total_cnt = 0

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()

        infer_result_dict = dict()

        # Number of buckets
        M = 20
        bucket_limits = torch.linspace(0, 1, M + 1)
        bucket_data = {'accuracies': torch.zeros(M), 'confidences': torch.zeros(M), 'counts': torch.zeros(M)}

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data[0].cuda()
            labels = data[-1].cuda()

            if len(data_loader.dataset.classes) < 200:
                for sample_idx in range(x.shape[0]):
                    tgt = gt[sample_idx].item()
                    class_name = data_loader.dataset.classes[tgt]
                    id = full_cub_dataset.class_to_idx[class_name]
                    gt[sample_idx] = id

            if PRODUCT_OF_EXPERTS is True:
                ########################################################
                out = MODEL1(x)
                out = out.cpu().detach()
                model1_p = torch.nn.functional.softmax(out, dim=1)

                score, index = torch.topk(model1_p, 1, dim=1)

                # Top-1 prediction is the first element in each row of 'index'
                top1_pred = index[:, 0]

                # Increment correct count if prediction matches label
                correct_cnt_model1 += (top1_pred == gt).sum().item()
                ########################################################

            # score, index = torch.topk(model1_p, depth_of_pred, dim=1)

            output_tensors = []
            depth = data[1].shape[1]
            # Loop to get the logit for each class
            for class_idx in range(data[1].shape[1]):
                explanation = data[1][:, class_idx, :, :, :, :]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                output, _, _, _ = model(images=x, explanations=explanation, scores=None)
                output = output.squeeze()
                output_tensors.append(output)

            logits = torch.stack(output_tensors, dim=1)
            # convert logits to probabilities using softmax function
            p = torch.softmax(logits, dim=1)

            if PRODUCT_OF_EXPERTS is True:
                conf_score_advnet = torch.sigmoid(logits)
                score_topk, index_topk = torch.topk(model1_p, data[1].shape[1], dim=1)
                final_confidence_score = conf_score_advnet.cpu()*score_topk
                score, final_preds = torch.topk(final_confidence_score, 1, dim=1)
                index = index_topk[torch.arange(index_topk.size(0)), final_preds.squeeze()]
            else:
                # Compute top-1 predictions and accuracy
                score, index = torch.topk(p, 1, dim=1)
                index = labels[torch.arange(len(index)), index.flatten()]
            index = index.cpu()

            running_corrects += torch.sum(index.squeeze() == gt)
            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))
            print(f'Depth of prediction: {depth}')

            if PRODUCT_OF_EXPERTS is True:
                # Calculate top-1 accuracy
                top1_accuracy = correct_cnt_model1 / total_cnt * 100
                print(f"Top-1 {RunningParams.set} accuracy of the base Classifier C: {top1_accuracy:.2f}%")

            # For ECE calculation: Get the maximum predicted probability and its corresponding label
            max_probs, preds = score, index
            correct = preds.eq(gt).float()

            max_probs = max_probs.detach().cpu()
            preds = preds.detach().cpu()
            correct = correct.detach().cpu()
            correct = correct.unsqueeze(1)

            # Sort the confidences into buckets
            for i in range(M):
                in_bucket = max_probs.gt(bucket_limits[i]) & max_probs.le(bucket_limits[i + 1])
                bucket_data['counts'][i] += in_bucket.float().sum()
                bucket_data['accuracies'][i] += (in_bucket.float() * correct).sum()
                bucket_data['confidences'][i] += (in_bucket.float() * max_probs).sum()

                # breakpoint()

        # Calculate the average accuracy and confidence for each bucket
        for i in range(M):
            if bucket_data['counts'][i] > 0:
                bucket_data['accuracies'][i] /= bucket_data['counts'][i]
                bucket_data['confidences'][i] /= bucket_data['counts'][i]

        # breakpoint()
        # Calculate ECE
        ece = 0.0
        for i in range(M):
            ece += (bucket_data['counts'][i] / total_cnt) * torch.abs(
                bucket_data['accuracies'][i] - bucket_data['confidences'][i])

        print(total_cnt)
        print("Expected Calibration Error (ECE): {:.4f}".format(100*ece.item()))

        import matplotlib.pyplot as plt

        # Assuming bucket_data is already computed as in the provided code snippet
        # We will create a similar plot to the uploaded one using the calculated ECE and bucket data

        # Extract the average accuracy and average confidence for each bucket
        accuracies = bucket_data['accuracies'].cpu().numpy()
        confidences = bucket_data['confidences'].cpu().numpy()
        counts = bucket_data['counts'].cpu().numpy()
        total_count = counts.sum()

        # Normalize counts to get the proportion of data points in each bucket
        proportions = counts / total_count

        # Create the reliability diagram
        fig, ax = plt.subplots(figsize=(6, 6))

        # Add the bars indicating the confidence for each bin
        ax.bar(confidences, accuracies, width=1.0 / M, edgecolor='black', alpha=0.5, label='Outputs')

        # Add the identity line
        ax.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')

        # Annotate ECE on the plot
        ece_percentage = 100 * ece.item()  # ECE as a percentage
        # ax.text(0.1, 0.9, f'ECE: {ece_percentage:.2f}%', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Set the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')

        # Set the title and legend
        ax.set_title(f'Reliability Diagram: ECE: {ece_percentage:.2f}%')
        ax.legend()

        plt.show()
        plt.savefig(f'{RunningParams.prj_dir}/Reliability_AdvNet_Diagram_test_top1_HP_MODEL1_HP_FE.png')


