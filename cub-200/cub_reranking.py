# Run advising process using AdvNet
import torch
import torch.nn as nn
import os
import argparse

import sys
sys.path.insert(0, '/home/giang/Downloads/advising_network')


from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs, ImageFolderWithPaths
from transformer import Transformer_AdvisingNetwork, CNN_AdvisingNetwork

RunningParams = RunningParams('CUB')

Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

torch.manual_seed(42)

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/RunningParams.combined_path',
                                     Dataset.data_transforms['train'])

PRODUCT_OF_EXPERTS = True
MODEL1_RESNET = True

depth = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_' + RunningParams.wandb_sess_name + '.pt',
                        # default='best_model_serene-field-3176.pt',
                        # default='best_model_cub_rn50_bs256-p1.0-dropout-0.0-NNth-1.pt', # RN50 ImageNet pretrained CUB
                        # default='best_model_cub_rn34_bs256-p1.0-dropout-0.0-NNth-1.pt', # RN34 ImageNet pretrained CUB
                        # default='best_model_cub_rn18_bs256-p1.0-dropout-0.0-NNth-1.pt', # RN18 ImageNet pretrained CUB
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

                        # default='best_model_legendary-durian-3216.pt', # Random negative samples data sampling, RN50, 1st NN

                        # default='best_model_genial-plasma-3125.pt',
                        default='best_model_decent-pyramid-3156.pt', # Normal model, top10, run1
                        # default='best_model_eager-field-3187.pt',  # Normal model, top10, run2
                        # default='best_model_light-cosmos-3188.pt',  # Normal model, top10, run3
                        # default='best_model_faithful-rain-3211.pt',
                        # default='best_model_legendary-durian-3216.pt',  # M=N=L=1 model convs only, no SA
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)
    correct_cnt_model1 = 0

    if MODEL1_RESNET is True:
        ################################################################################
        from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck
        from torchvision import datasets, models, transforms

        resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load(
            f'{RunningParams.prj_dir}/pretrained_models/cub-200/iNaturalist_pretrained_RN50_85.83.pth')

        if RunningParams.resnet == 50 and RunningParams.RN50_INAT is False:
            resnet = models.resnet50(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(2048, 200))
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet50_cub_200way_top1acc_63.pth')
        elif RunningParams.resnet == 34:
            resnet = models.resnet34(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(512, 200))
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet34_cub_200way_top1acc_62_81.pth')
        elif RunningParams.resnet == 18:
            resnet = models.resnet18(pretrained=True)
            resnet.fc = nn.Sequential(nn.Linear(512, 200))
            my_model_state_dict = torch.load(
                f'{RunningParams.prj_dir}/pretrained_models/cub-200/imagenet_pretrained_resnet18_cub_200way_top1acc_60_22.pth')

        resnet.load_state_dict(my_model_state_dict, strict=True)
        if RunningParams.resnet == 34 or RunningParams.resnet == 18 or (
                RunningParams.resnet == 50 and RunningParams.RN50_INAT is False):
                resnet.fc = resnet.fc[0]

        # MODEL1 = resnet
        # MODEL1 = nn.DataParallel(MODEL1).cuda()
        # MODEL1.eval()

        ############ ViTB-16 ################
        # Initialize the base model and load the trained weights
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


        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=200)
        model_path_vit = f"{RunningParams.prj_dir}/pretrained_models/cub-200/vit_base_patch16_224_cub_200way_82_40.pth"
        state_dict = torch.load(model_path_vit)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(new_state_dict)

        # Wrap the base model in the custom model
        model = CustomViT(base_model).cuda()
        model.eval()

        MODEL1 = nn.DataParallel(model).cuda()

        #####################################

        ###############################################################################
    else:
        if RunningParams.NTSNET is True: # NTSNet
            ################################################################
            import os
            from torch.autograd import Variable
            import torch.utils.data
            from torch.nn import DataParallel
            from core import model, dataset
            from torch import nn
            from torchvision.datasets import ImageFolder
            from tqdm import tqdm

            net = model.attention_net(topN=6)
            ckpt = torch.load(f'{RunningParams.parent_dir}/NTS-Net/model.ckpt')

            net.load_state_dict(ckpt['net_state_dict'])

            net.eval()
            net = net.cuda()
            net = DataParallel(net)
            MODEL1 = net

            from torchvision import transforms
            from PIL import Image

            ntsnet_data_transforms = transforms.Compose([
                transforms.Resize((600, 600), interpolation=Image.BILINEAR),
                transforms.CenterCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # data_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/{}'.format(set)
            data_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'
            nts_val_data = ImageFolderWithPaths(
                # ImageNet train folder
                root=data_dir, transform=ntsnet_data_transforms
            )

            val_data = ImageFolderWithPaths(
                # ImageNet train folder
                root=data_dir, transform=Dataset.data_transforms['val']
            )

            nts_train_loader = torch.utils.data.DataLoader(
                nts_val_data,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False
            )

            train_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False
            )
            print('Running MODEL1 being NTS-NET!!!')

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

    print(f'PRODUCT_OF_EXPERTS: {PRODUCT_OF_EXPERTS}')

    model.eval()

    test_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'

    image_datasets = dict()
    if MODEL1_RESNET is True:
        image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    else:
        if RunningParams.NTSNET is True:
            image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, ntsnet_data_transforms)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=5,
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

            # if PRODUCT_OF_EXPERTS is True:
            if True:
                ########################################################
                if MODEL1_RESNET is False and RunningParams.NTSNET is True:
                    _, out, _, _, _ = MODEL1(data[2].cuda())
                else:
                    out, _ = MODEL1(x)

                out = out.cpu().detach()
                model1_p = torch.nn.functional.softmax(out, dim=1)

                score, index_top1 = torch.topk(model1_p, 1, dim=1)

                correct_predictions = index_top1.squeeze(1) == gt
                correct_cnt_model1 += correct_predictions.sum().item()

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

            # if PRODUCT_OF_EXPERTS is True:
            if True:
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

        # Calculate the average accuracy and confidence for each bucket
        for i in range(M):
            if bucket_data['counts'][i] > 0:
                bucket_data['accuracies'][i] /= bucket_data['counts'][i]
                bucket_data['confidences'][i] /= bucket_data['counts'][i]

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


