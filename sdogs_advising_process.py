import torch
import torch.nn as nn
import os
import argparse

from tqdm import tqdm
from params import RunningParams
from datasets import Dataset, ImageFolderForAdvisingProcess, StanfordDogsDataset
from transformer import Transformer_AdvisingNetwork

RunningParams = RunningParams()
Dataset = Dataset()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.manual_seed(42)

from torchvision import datasets, models, transforms

import torchvision.transforms as T
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

# Because it has a unique type of mapping the labels, we need reference_dataset to convert the predicted ids (StanfordDogsDataset) to ImageFolder ids
reference_dataset = StanfordDogsDataset(
    root=os.path.join('/home/giang/Downloads/advising_network/stanford-dogs', "data"), set_type="train", transform=preprocess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # default='best_model_serene-bush-3128.pt',
                        # default='best_model_stoic-water-3131.pt',
                        default='best_model_eager-cloud-3132.pt',
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
    print(epoch)

    print('Validation accuracy: {:.4f}'.format(acc))
    print('F1 score: {:.4f}'.format(f1))

    model.eval()

    test_dir = '/home/giang/Downloads/advising_network/stanford-dogs/data/images/test'  ##################################

    image_datasets = dict()
    image_datasets['cub_test'] = ImageFolderForAdvisingProcess(test_dir, preprocess)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['cub_test']}

    for ds in ['cub_test']:
        data_loader = torch.utils.data.DataLoader(
            image_datasets[ds],
            batch_size=12,
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
                    id = reference_dataset.class_to_idx[class_name]
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
