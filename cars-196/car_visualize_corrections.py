# Visualize AdvNet corrections after re-ranking

import torch
import torch.nn as nn
import os
import argparse

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams()
Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.manual_seed(42)

from torchvision import datasets, models, transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/RunningParams.train_path',
                                     data_transform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_zesty-mountain-3152.pt',
                        # default='best_model_spring-field-3157.pt',
                        default='best_model_robust-sunset-3158.pt',
                        # default='best_model_divine-cherry-3160.pt',
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)
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
    f1 = checkpoint['best_f1']

    print('best model found at epoch {}'.format(epoch))

    print('Validation accuracy: {:.4f}'.format(acc))
    print('F1 score: {:.4f}'.format(f1))

    model.eval()

    test_dir = f'{RunningParams.parent_dir}/RunningParams.test_path'  ##################################

    import numpy as np

    file_name = f'{RunningParams.prj_dir}/faiss/advising_process_top1_Cars.npy'
    faiss_nn_dict = np.load(file_name, allow_pickle=True, ).item()

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    import random

    random.seed(42)
    np.random.seed(42)
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
        running_corrects_top5 = 0
        total_cnt = 0

        yes_cnt = 0
        true_cnt = 0
        confidence_dict = dict()

        correction_result_dict = dict()
        cnt = 0

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data[0].cuda()
            labels = data[-1].cuda()

            if len(data_loader.dataset.classes) < 120:
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

                # print(explanation.shape)
                output, _, _, _ = model(images=x, explanations=explanation, scores=model1_score)
                output = output.squeeze()
                output_tensors.append(output)

            logits = torch.stack(output_tensors, dim=1)
            # convert logits to probabilities using softmax function
            p = torch.softmax(logits, dim=1)
            p_sigmoid = torch.sigmoid(logits)

            # # Compute top-1 predictions and accuracy
            # score, index = torch.topk(p, 1, dim=1)
            # index = labels[torch.arange(len(index)), index.flatten()]

            # Get the sorted indices along each row
            # _, indices = torch.sort(logits, dim=1, descending=True)
            #
            # # Now use these indices to rearrange each row of labels
            # refined_labels = labels.gather(1, indices)

            for sample_idx in range(x.shape[0]):
                path = pths[sample_idx]
                base_name = os.path.basename(path)
                original_preds = labels[sample_idx]
                sim_scores = p_sigmoid[sample_idx]

                # breakpoint()

                nn_dict = faiss_nn_dict[base_name]
                model1_scores = torch.tensor([nn_dict[i]['C_confidence'] for i in range(len(nn_dict))])
                poe_score = model1_scores * sim_scores.cpu()
                sim_scores, indices = torch.sort(poe_score, dim=0, descending=True)
                refined_preds = original_preds[indices]

                nns = list()
                for k, v in nn_dict.items():
                    nns.append(v['NNs'][0])
                # breakpoint()
                # If the new top1 matches the GT  && If the new top1 is different from the old top 1
                if refined_preds[0].item() == gt[sample_idx].item() and \
                        original_preds[0].item() != refined_preds[0].item():

                    import matplotlib.pyplot as plt
                    from PIL import Image
                    import numpy as np
                    import subprocess
                    import os


                    # Function to resize and crop the image
                    def resize_and_crop(image):
                        image = image.resize((256, 256))
                        width, height = image.size
                        left = (width - 224) / 2
                        top = (height - 224) / 2
                        right = (width + 224) / 2
                        bottom = (height + 224) / 2
                        image = image.crop((left, top, right, bottom))
                        return image


                    # Convert the tensors to lists
                    original_preds = original_preds.tolist()
                    refined_preds = refined_preds.tolist()
                    sim_scores = sim_scores.tolist()

                    # Prepare figure and axes, increase the figsize to make sub-images larger
                    fig, axs = plt.subplots(1, 6, figsize=(30, 5))
                    fig.subplots_adjust(wspace=0.15, hspace=0.3)
                    fig.suptitle('Initial class ranking by pretrained classifier C', color='red',
                                 size=20)  # Add this line

                    # Load and plot the original image
                    original_img = Image.open(path)
                    original_img = resize_and_crop(original_img)
                    axs[0].imshow(np.array(original_img))
                    axs[0].set_title('Query: {}'.format(data_loader.dataset.classes[gt[sample_idx].item()]),
                                     color='green', fontsize=12)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])

                    # For each original prediction, load the corresponding image, plot it, and show the similarity score
                    for i, pred in enumerate(original_preds):
                        pred_img = Image.open(nns[i])
                        pred_img = resize_and_crop(pred_img)
                        axs[i + 1].imshow(np.array(pred_img))
                        # axs[i + 1].set_title(
                        #     f'Top{i + 1} {data_loader.dataset.classes[pred]}, Confidence: {sim_scores[i]:.2f}')

                        if data_loader.dataset.classes[pred] == data_loader.dataset.classes[gt[sample_idx].item()]:
                            color = 'green'
                        else:
                            color = 'black'

                        # Set the title for the plot (at the top by default)
                        axs[i + 1].set_title(f'Top{i + 1}: {data_loader.dataset.classes[pred]}', color=color,
                                             fontsize=12)

                        # Add the confidence at the bottom of the image
                        # axs[i + 1].text(0.5, -0.07, f'AdvNet\'s Confidence: {sim_scores[i]:.2f}', size=18, ha="center",
                        #                 transform=axs[i + 1].transAxes)

                        conf = nn_dict[i]['C_confidence']
                        sim = p_sigmoid[sample_idx][i]
                        axs[i + 1].text(0.5, -0.07, f'RN50: {int(conf * 100)}% | S: {sim:.2f}', size=18,
                                        ha="center",
                                        transform=axs[i + 1].transAxes)

                        axs[i + 1].set_xticks([])
                        axs[i + 1].set_yticks([])

                    # Save the figure before clear
                    plt.savefig('before.jpeg', bbox_inches='tight', pad_inches=0)  # reduced padding in saved figure
                    plt.close()

                    # Repeat the same steps for the refined predictions
                    fig, axs = plt.subplots(1, 6, figsize=(30, 5))
                    fig.subplots_adjust(wspace=0.15, hspace=0.3)
                    fig.suptitle('Refined class ranking by Product of Experts C x S', color='green',
                                 size=20)  # Add this line

                    # Load the original image
                    original_img = Image.open(path)
                    original_img = resize_and_crop(original_img)

                    # Create a white image with the same size
                    white_img = Image.fromarray(
                        np.full((original_img.size[1], original_img.size[0], 3), 255).astype(np.uint8))

                    # Plot the white image
                    axs[0].imshow(np.array(white_img))
                    axs[0].axis('off')  # This removes the border around the image
                    # axs[0].set_xticks([])
                    # axs[0].set_yticks([])

                    # For each refined prediction, load the corresponding image and plot it
                    for i, pred in enumerate(refined_preds):
                        pred_img = Image.open(nns[original_preds.index(pred)])
                        pred_img = resize_and_crop(pred_img)
                        axs[i + 1].imshow(np.array(pred_img))

                        if data_loader.dataset.classes[pred] == data_loader.dataset.classes[gt[sample_idx].item()]:
                            color = 'green'
                        else:
                            color = 'black'
                            # Set the title for the plot (at the top by default)
                        axs[i + 1].set_title(f'Top{i + 1}: {data_loader.dataset.classes[pred]}', color=color,
                                             fontsize=12)

                        sim_scores = sorted(sim_scores, reverse=True)

                        axs[i + 1].text(0.5, -0.07, f'RN50 x S: {int(sim_scores[i] * 100)}%', size=18, ha="center",
                                        transform=axs[i + 1].transAxes)

                        axs[i + 1].set_xticks([])
                        axs[i + 1].set_yticks([])

                    # Save the figure before clear
                    plt.savefig('after.jpeg', bbox_inches='tight', pad_inches=0)  # reduced padding in saved figure
                    plt.close()

                    subprocess.call(
                        'montage before.jpeg after.jpeg -tile 1x2 -geometry +20+20 {}/corrections/cars/{}_{}_{}.jpeg'.
                        format(RunningParams.prj_dir, data_loader.dataset.classes[gt[sample_idx].item()].replace(' ', '_'), batch_idx,
                               sample_idx),
                        shell=True)

                    jpeg_path = '{}/corrections/cars/{}_{}_{}.jpeg'.format(
                        RunningParams.prj_dir, data_loader.dataset.classes[gt[sample_idx].item()].replace(' ', '_'), batch_idx, sample_idx)
                    pdf_path = jpeg_path.replace('.jpeg', '.pdf')

                    subprocess.call('convert {} {}'.format(jpeg_path, pdf_path), shell=True)
                    os.remove(jpeg_path)

            # running_corrects += torch.sum(index.squeeze() == gt.cuda())
            # total_cnt += data[0].shape[0]
            #
            # print(cnt)
            # print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))
