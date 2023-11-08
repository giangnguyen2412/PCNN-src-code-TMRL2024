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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.manual_seed(42)

from torchvision import datasets, models, transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
full_cub_dataset = ImageFolderForNNs(f'{RunningParams.parent_dir}/Cars/Stanford-Cars-dataset/train',
                                     data_transform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_zesty-mountain-3152.pt',
                        # default='best_model_spring-field-3157.pt',
                        # default='best_model_robust-sunset-3158.pt',
                        # default='best_model_divine-cherry-3160.pt',
                        # default='best_model_wandering-capybara-3189.pt',  # RN50 run 2
                        default='best_model_different-lion-3192.pt',  # RN50 run 3
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
    f1 = checkpoint['best_f1']

    print('Validation accuracy: {:.4f}'.format(acc))
    print('F1 score: {:.4f}'.format(f1))

    model.eval()

    test_dir = f'{RunningParams.parent_dir}/Cars/Stanford-Cars-dataset/test'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, Dataset.data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=16,
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

            # Compute top-1 predictions and accuracy
            score, index = torch.topk(p, 1, dim=1)
            index = labels[torch.arange(len(index)), index.flatten()]

            running_corrects += torch.sum(index.squeeze() == gt.cuda())
            total_cnt += data[0].shape[0]

            print("Top-1 Accuracy: {}".format(running_corrects * 100 / total_cnt))
