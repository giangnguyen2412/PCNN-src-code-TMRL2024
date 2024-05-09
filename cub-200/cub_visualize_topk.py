# Visualize PoE CxS top-k predictions
import torch
import torch.nn as nn
import os
import argparse
import json

import sys
sys.path.append('/home/giang/Downloads/advising_network')

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, ImageFolderForNNs
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams('CUB')

Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/{RunningParams.combined_path}',
                                     Dataset.data_transforms['train'])

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import subprocess
import os

TOP1_NN = False
PRODUCT_OF_EXPERTS = True  # We already store the confidence scores of the C classifier ResNet in the faiss dict

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

import matplotlib.backends.backend_pdf

correct_figures = []
incorrect_figures = []
sample_num = 300
# Dictionary to track the correctness of each figure
correctness_dict = {}
metadata = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_genial-plasma-3125.pt',
                        default='best_model_decent-pyramid-3156.pt',
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join(RunningParams.prj_dir, 'best_models', args.ckpt)
    print(args)

    model = Transformer_AdvisingNetwork()
    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(model_path)
    running_params = checkpoint['running_params']

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    acc = checkpoint['val_acc']

    f1 = checkpoint['best_f1']
    print('best model found at epoch {}'.format(epoch))

    print('Validation accuracy: {:.4f}'.format(acc))
    print('F1 score: {:.4f}'.format(f1))

    model.eval()

    # test_dir = f'{RunningParams.parent_dir}/datasets/CUB/advnet/test'  ##################################
    test_dir = f'{RunningParams.parent_dir}/{RunningParams.test_path}'

    import numpy as np
    file_name = f'{RunningParams.prj_dir}/faiss/advising_process_test_top1_HP_MODEL1_HP_FE.npy'
    faiss_nn_dict = np.load(file_name, allow_pickle=True, ).item()

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    import random

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)

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
            p_sigmoid = torch.sigmoid(logits)

            # Compute top-1 predictions and accuracy
            score, index = torch.topk(p, 1, dim=1)
            index = labels[torch.arange(len(index)), index.flatten()]

            # Get the sorted indices along each row
            _, indices = torch.sort(logits, dim=1, descending=True)

            # Now use these indices to rearrange each row of labels
            refined_labels = labels.gather(1, indices)

            for sample_idx in range(x.shape[0]):

                # Check correctness of the prediction
                is_correct = index[sample_idx].item() == gt[sample_idx].item()

                # Proceed with visualization only if needed
                if (is_correct and len(correct_figures) < sample_num) or (not is_correct and len(incorrect_figures) < sample_num):

                    path = pths[sample_idx]
                    base_name = os.path.basename(path)
                    original_preds = labels[sample_idx]
                    sim_scores = p_sigmoid[sample_idx]
                    refined_preds = refined_labels[sample_idx]
                    nn_dict = faiss_nn_dict[base_name]

                    if PRODUCT_OF_EXPERTS is True:
                        advnet_confs = sim_scores.detach().cpu()
                        c_confs = torch.tensor([nn_dict[i]['C_confidence'] for i in range(data[1].shape[1])])
                        final_confidence_score = advnet_confs*c_confs
                        score, final_preds = torch.topk(final_confidence_score, data[1].shape[1], dim=0)
                        # TODO: Normalize score using sofmmax if required

                    nns = list()
                    for k, v in nn_dict.items():
                        nns.append(v['NNs'][0])

                    if TOP1_NN is True:
                        id = final_preds[0].item()
                        top1_data = nn_dict[id]
                        top1_nns = top1_data['NNs']

                    # Convert the tensors to lists
                    # original_preds = original_preds.tolist()
                    # refined_preds = refined_preds.tolist()
                    # sim_scores = sim_scores.tolist()
                    # breakpoint()

                    if PRODUCT_OF_EXPERTS is True:
                        final_preds = original_preds[final_preds]
                    else:
                        final_preds = refined_preds

                    original_preds = original_preds.tolist()

                    # Prepare figure and axes, increase the figsize to make sub-images larger
                    fig, axs = plt.subplots(1, 6, figsize=(30, 5), gridspec_kw={'width_ratios': [1.5, 1, 1, 1, 1, 1]})
                    fig.subplots_adjust(wspace=0.03, hspace=0.3)
                    # fig.suptitle(f'Visual explanation by showing top{data[1].shape[1]}-predicted-class training nearest neighbors', color='blue', size=20)  # Add this line

                    # Load and plot the original image
                    original_img = Image.open(path)
                    original_img = resize_and_crop(original_img)
                    axs[0].imshow(np.array(original_img))
                    # axs[0].set_title(f'Input {batch_idx}-{sample_idx}', fontsize=18)
                    axs[0].set_title(f'Input', fontsize=18)
                    # axs[0].set_title('Query: {}'.format(data_loader.dataset.classes[gt[sample_idx].item()].split('.')[1].replace('_',' ')), color='green', fontsize=18)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])

                    # For each original prediction, load the corresponding image, plot it, and show the similarity score
                    for i, pred in enumerate(final_preds):
                        if TOP1_NN is True:
                            pred_img = Image.open(top1_nns[i])
                        else:
                            pred_img = Image.open(nns[original_preds.index(pred)])
                        pred_img = resize_and_crop(pred_img)
                        axs[i + 1].imshow(np.array(pred_img))

                        class_name = data_loader.dataset.classes[pred].split('.')[1].replace('_',' ')
                        color = 'black'
                        if i == 0:
                            class_name_for_question = class_name
                            score_for_question = score[i].item()

                        # Set the title for the plot (at the top by default)

                        if TOP1_NN is True:
                            axs[i + 1].set_title(f'{class_name_for_question}', color=color, fontsize=14)
                        else:
                            # axs[i + 1].set_title(f'Top{i + 1}: {class_name}', color=color, fontsize=14)
                            axs[i + 1].set_title(f'{class_name}', color=color, fontsize=14)


                        axs[i + 1].set_xticks([])
                        axs[i + 1].set_yticks([])

                    # Store the figure object based on correctness
                    if is_correct:
                        correct_figures.append(fig)
                    else:
                        incorrect_figures.append(fig)

                    # breakpoint()
                    # Save the figure before clear
                    class_name = data_loader.dataset.classes[gt[sample_idx].item()]
                    save_path = f'{RunningParams.prj_dir}/corrections/cub/{class_name}_{batch_idx}_{sample_idx}.jpeg'
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    correctness_dict[f'{batch_idx}-{sample_idx}'] = is_correct
                    # metadata
                    metadata[f'{class_name}_{batch_idx}_{sample_idx}'] = {}
                    metadata[f'{class_name}_{batch_idx}_{sample_idx}']['top1-label'] = class_name_for_question
                    metadata[f'{class_name}_{batch_idx}_{sample_idx}']['top1-score'] = score_for_question
                    metadata[f'{class_name}_{batch_idx}_{sample_idx}']['question'] = \
                        f'Sam guessed the Input image is {int(score_for_question*100)}% {class_name_for_question}. Is it {class_name_for_question}?'
                    metadata[f'{class_name}_{batch_idx}_{sample_idx}']['Accept/Reject'] = is_correct

                if len(correct_figures) >= sample_num and len(incorrect_figures) >= sample_num:
                    break

            if len(correct_figures) >= sample_num and len(incorrect_figures) >= sample_num:
                break

        correctness_file_path = f'{RunningParams.prj_dir}/corrections/cub/correctness_info.json'
        with open(correctness_file_path, 'w') as json_file:
            json.dump(correctness_dict, json_file)

        metadata_path = f'{RunningParams.prj_dir}/corrections/cub/metadata.json'
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file)

        # Function to save figures into a PDF
        def save_figures_to_pdf(figures, pdf_path):
            with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
                for fig in figures:
                    pdf.savefig(fig, bbox_inches='tight')

        random.seed(100)
        figures = correct_figures[:sample_num] + incorrect_figures[:sample_num]
        random.shuffle(figures)
        # save_figures_to_pdf(figures, f'{RunningParams.prj_dir}/predictions.pdf')

        running_corrects += torch.sum(index.squeeze() == gt.cuda())
        total_cnt += data[0].shape[0]

        print(cnt)
        print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))
