import torch
import torch.nn as nn
import os
import argparse

import sys
sys.path.insert(0, '/home/giang/Downloads/advising_network')


from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs
from transformer import Transformer_AdvisingNetwork, CNN_AdvisingNetwork

RunningParams = RunningParams()
Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

torch.manual_seed(42)

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/datasets/CUB/combined',
                                     Dataset.data_transforms['train'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_serene-field-3176.pt',
                        # default='best_model_cosmic-waterfall-3174.pt',  # no CA --> have no cross attn to visualize
                        # default='best_model_young-planet-3170.pt',  # no SA --> clear heatmaps
                        # default='best_model_avid-cosmos-3201.pt',  # M=N=L=1 model
                        # default='best_model_serene-sound-3240.pt',  # M=N=L=2 model
                        # default='best_model_misty-sky-3239.pt',  # 1st NNs
                        # default='best_model_blooming-sponge-3236.pt',  # 2nd NNs
                        # default='best_model_lilac-waterfall-3238.pt',  # 3rd NNs
                        # default='best_model_lilac-bird-3237.pt',  # 5th NNs

                        # default='best_model_different-grass-3212.pt',
                        # default='best_model_bs256-p1.0-dropout-0.0-NNth-3.pt',

                        # default='best_model_genial-plasma-3125.pt',
                        default='best_model_decent-pyramid-3156.pt',
                        # default='best_model_eager-field-3187.pt',  # Normal model, top10, run2
                        # default='best_model_light-cosmos-3188.pt',  # Normal model, top10, run3
                        # default='best_model_faithful-rain-3211.pt',
                        # default='best_model_legendary-durian-3216.pt',  # M=N=L=1 model convs only, no SA
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)

    print(args)

    if RunningParams.TRANSFORMER_ARCH == True:
        MODEL2 = Transformer_AdvisingNetwork()
    else:
        MODEL2 = CNN_AdvisingNetwork()
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

    model.eval()

    # test_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/test'  ##################################
    test_dir = f'{RunningParams.parent_dir}/datasets/CUB/test0'


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

            # Make a dummy confidence score
            model1_score = torch.zeros([data[1].shape[0], 1]).cuda()

            output_tensors = []
            # Loop to get the logit for each class
            for class_idx in range(data[1].shape[1]):
                explanation = data[1][:, class_idx, :, :, :, :]
                explanation = explanation[:, 0:RunningParams.k_value, :, :, :]

                output, _, _, _ = model(images=x, explanations=explanation, scores=model1_score)
                output = output.squeeze()
                output_tensors.append(output)

            logits = torch.stack(output_tensors, dim=1)
            # convert logits to probabilities using softmax function
            p = torch.softmax(logits, dim=1)

            # Compute top-1 predictions and accuracy
            score, index = torch.topk(p, 1, dim=1)
            index = labels[torch.arange(len(index)), index.flatten()]

            running_corrects += torch.sum(index.squeeze() == gt.cuda())
            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))

            # For ECE calculation: Get the maximum predicted probability and its corresponding label
            max_probs, preds = score, index
            correct = preds.eq(gt.cuda()).float()

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


