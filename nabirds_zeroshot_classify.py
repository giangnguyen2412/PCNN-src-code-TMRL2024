import torch
import torch.nn as nn
import os
import argparse

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForZeroshot
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams()
Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_divine-snowflake-2832.pt',
                        default='best_model_vague-rain-2946.pt',
                        help='Model check point')

    args = parser.parse_args()
    model_path = os.path.join('best_models', args.ckpt)
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

    print('Validation accuracy: {:.2f}'.format(acc))

    model.eval()

    # test_dir = '/home/giang/Downloads/nabirds_exclusive_split_small_50/test'  ##################################
    test_dir = '/home/giang/Downloads/Cars/Stanford-Cars-dataset-reduced/test'

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForZeroshot(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=8,
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

        for batch_idx, (data, gt, pths) in enumerate(tqdm(data_loader)):
            x = data[0].cuda()

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
            running_corrects += torch.sum(index.squeeze() == gt.cuda())

            # Compute top-5 predictions and accuracy
            score_top5, index_top5 = torch.topk(logits, 5, dim=1)
            gt_expanded = gt.cuda().unsqueeze(1).expand_as(index_top5)
            running_corrects_top5 += torch.sum(index_top5 == gt_expanded)

            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects*100/total_cnt))
            print("Top-5 Accuracy: {}".format(running_corrects_top5*100/total_cnt))