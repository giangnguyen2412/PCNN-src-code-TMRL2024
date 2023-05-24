import torch
import torchvision.transforms as transforms
import json
import os
import numpy as np
from params import RunningParams
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import cv2
import torchvision.transforms as T
from PIL import Image

# Define the TrivialAugmentWide transform
trivial_augmenter = T.TrivialAugmentWide()
jitter = T.ColorJitter(brightness=.5, hue=.3)

# Define the RandomApply transform to apply the TrivialAugmentWide transform with a probability of 0.5
trivial_augmenter = T.RandomApply(torch.nn.ModuleList([trivial_augmenter, jitter]), p=0.5)

RunningParams = RunningParams()


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __init__ method to drop no-label images
    if RunningParams.IMAGENET_REAL:
        def __init__(self, root, transform=None):
            super(ImageFolderWithPaths, self).__init__(root, transform=transform)

            if os.path.basename(root) == 'train':
                pass
            else:
                real_json = open("reassessed-imagenet/real.json")
                real_ids = json.load(real_json)
                real_labels = {
                    f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
                    for i, labels in enumerate(real_ids)
                }

                original_len = len(self.imgs)
                imgs = []
                samples = []
                targets = []
                for sample_idx in range(original_len):
                    pth = self.imgs[sample_idx][0]
                    base_name = os.path.basename(pth)
                    # Not ImageNet-val then we exit the function
                    # if base_name not in real_labels:
                    #     return
                    if RunningParams.IMAGENET_TRAINING:
                        real_ids = real_labels[base_name]
                        if len(real_ids) == 0:
                            continue

                    else:
                        imgs.append(self.imgs[sample_idx])
                        samples.append(self.samples[sample_idx])
                        targets.append(self.targets[sample_idx])

                self.imgs = imgs
                self.samples = samples
                self.targets = targets

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        data = original_tuple[0]  # --> 3x224x224 --> 7x3x224x224
        label = original_tuple[1]
        if data.shape[0] == 1:
            print('gray images')
            data = torch.cat([data, data, data], dim=0)

        # make a new tuple that includes original and the path
        tuple_with_path = (data, label, path)
        return tuple_with_path


class ImageFolderForNNs(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __init__ method to drop no-label images

    if RunningParams.IMAGENET_REAL:
        def __init__(self, root, transform=None):
            super(ImageFolderForNNs, self).__init__(root, transform=transform)

            self.root = root
            # Load the pre-computed NNs
            if RunningParams.CUB_TRAINING is True:
                if RunningParams.TOP1_NN is True:
                    if 'train' in os.path.basename(root):
                        if RunningParams.MODEL2_FINETUNING is True:
                            # file_name = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_train_top1_HP_MODEL1_HP_FE.npy'
                            # file_name = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_train_aug_top1_HP_MODEL1_HP_FE.npy'
                            # file_name = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_train_all_top1_HP_MODEL1_HP_FE.npy'
                            file_name = 'faiss/cub/top5_NeurIPS_Finetuning_faiss_CUB_train_all_top1_HP_MODEL1_HP_FE.npy'
                        else:  # Pretraining
                            if RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
                                file_name = 'faiss/cub/NeurIPS_Pretraining_faiss_CUB_CUB_pre_train_top1_LP_MODEL1_HP_FE.npy'
                    else:
                        if RunningParams.MODEL2_FINETUNING is True:
                            if 'val' in os.path.basename(root):
                                file_name = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_val_top1_HP_MODEL1_HP_FE.npy'
                            else:
                                file_name = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_test_top1_HP_MODEL1_HP_FE.npy'
                        else:  # Pretraining
                            if RunningParams.HIGHPERFORMANCE_FEATURE_EXTRACTOR is True:
                                if 'val' in os.path.basename(root):
                                    file_name = 'faiss/cub/NeurIPS_Pretraining_faiss_CUB_CUB_pre_val_top1_LP_MODEL1_HP_FE.npy'
                                else:  # dummy feature
                                    file_name = 'faiss/cub/NeurIPS_Pretraining_faiss_CUB_CUB_pre_val_top1_LP_MODEL1_HP_FE.npy'

            else:
                if RunningParams.TOP1_NN is True:
                    if 'train' in os.path.basename(root):
                        # file_name = 'faiss/faiss_SDogs_train_RN34_top1.npy'
                        file_name = 'faiss/faiss_SDogs_train_augment_RN34_top1.npy'
                    elif 'val' in os.path.basename(root):
                        file_name = 'faiss/faiss_SDogs_val_RN34_top1.npy'
                    elif 'test' in os.path.basename(root):
                        file_name = 'faiss/faiss_SDogs_test_RN34_top1.npy'
                    else:
                        exit(-1)

            print(file_name)
            self.faiss_nn_dict = np.load(file_name, allow_pickle=True, ).item()

            print(len(self.faiss_nn_dict))
            if os.path.basename(root) == 'train':
                pass
            else:
                real_json = open("reassessed-imagenet/real.json")
                real_ids = json.load(real_json)
                real_labels = {
                    f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
                    for i, labels in enumerate(real_ids)
                }

                original_len = len(self.imgs)
                imgs = []
                samples = []
                targets = []
                for sample_idx in range(original_len):
                    pth = self.imgs[sample_idx][0]
                    base_name = os.path.basename(pth)
                    # Not ImageNet-val then we exit the function
                    # if base_name not in real_labels:
                    #     return
                    if RunningParams.IMAGENET_TRAINING:
                        real_ids = real_labels[base_name]
                        if len(real_ids) == 0:
                            continue

                    else:
                        imgs.append(self.imgs[sample_idx])
                        samples.append(self.samples[sample_idx])
                        targets.append(self.targets[sample_idx])

                self.imgs = imgs
                self.samples = samples
                self.targets = targets

    def __getitem__(self, index):
        query_path, target = self.samples[index]
        base_name = os.path.basename(query_path)
        if RunningParams.XAI_method == RunningParams.NNs:
            if 'train' in os.path.basename(self.root):
                nns = self.faiss_nn_dict[base_name]['NNs']  # 6NNs here
                model2_target = self.faiss_nn_dict[base_name]['label']
            else:
                nns = self.faiss_nn_dict[base_name]  # 6NNs here

            # Transform NNs
            explanations = list()
            dup = False
            for pth in nns:
                sample = self.loader(pth)
                nn_base_name = os.path.basename(pth)
                if nn_base_name in base_name:
                    dup = True
                    continue
                sample = self.transform(sample)
                explanations.append(sample)
            # If query is the same with any of NNs --> duplicate the last element
            if dup is True:
                explanations += [explanations[-1]]
            explanations = torch.stack(explanations)

            # Transform query
            sample = self.loader(query_path)
            query = self.transform(sample)

            # make a new tuple that includes original and the path
            if 'train' in os.path.basename(self.root):
                tuple_with_path = ((query, explanations, model2_target), target, query_path)
            else:
                tuple_with_path = ((query, explanations), target, query_path)

            return tuple_with_path


class ImageFolderForZeroshot(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root, transform=None):
        super(ImageFolderForZeroshot, self).__init__(root, transform=transform)

        self.root = root
        # Load the pre-computed NNs
        if 'test' in os.path.basename(root):
            file_name = 'faiss/NN_dict_NA-Birds-small-zero-shot_test.npy'
        else:
            exit(-1)

        print(file_name)
        self.faiss_nn_dict = np.load(file_name, allow_pickle=True, ).item()

        original_len = len(self.imgs)
        imgs = []
        samples = []
        targets = []
        for sample_idx in range(original_len):
            imgs.append(self.imgs[sample_idx])
            samples.append(self.samples[sample_idx])
            targets.append(self.targets[sample_idx])

        self.imgs = imgs
        self.samples = samples
        self.targets = targets

    def __getitem__(self, index):
        query_path, target = self.samples[index]

        # Transform query
        sample = self.loader(query_path)
        query = self.transform(sample)

        base_name = os.path.basename(query_path)

        nns = self.faiss_nn_dict[base_name]  # 6NNs here

        # Initialize an empty tensor to store the transformed images
        tensor_images = torch.empty((len(nns), 6, 3, 224, 224))

        # Iterate over the dictionary entries and transform the images
        for i, key in enumerate(nns):
            file_paths = nns[key]
            for j, file_path in enumerate(file_paths):
                # Load the image using the loader function
                image = self.loader(file_path)  # Replace `loader` with your actual loader function

                # Apply the transformation to the image
                transformed_image = self.transform(image)

                # Assign the transformed image to the tensor
                tensor_images[i, j] = transformed_image

        tuple_with_path = ((query, tensor_images), target, query_path)

        return tuple_with_path


class Dataset(object):
    def __init__(self):
        # These datasets have < 1000 classes
        self.IMAGENET_A = "imagenet-a"
        self.OBJECTNET_5K = "objectnet-5k"
        self.IMAGENET_R = "imagenet-r"

        # These datasets have 1000 classes
        self.IMAGENET_1K = "balanced_val_dataset_6k"
        # self.IMAGENET_1K = "p_val"
        self.IMAGENET_1K_50K = "imagenet1k-val-50k"
        self.IMAGENET_1K_50K_CLEAN = "ImageNet-val-50K-clean"

        self.IMAGENET_SKETCH = "imagenet-sketch"
        self.DAMAGE_NET = "DAmageNet_processed"

        self.IMAGENET_HARD = "Hard"
        self.IMAGENET_MULTI_OBJECT = "Multi-Object"
        self.ADVERSARIAL_PATCH_NEW = "adversarial_patches"

        self.GAUSSIAN_NOISE = "gaussian_noise"
        self.GAUSSIAN_BLUR = "gaussian_blur"

        self.CUB200 = "cub200_test"

        self.IMAGENET_PILOT_VIS = "imagenet1k-pilot"

        self.test_datasets = [self.IMAGENET_1K]

        self.IMAGENET_C_NOISE = [self.GAUSSIAN_NOISE, self.GAUSSIAN_BLUR]

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if RunningParams.IMAGENET_REAL is True:
            real_json = open("/home/giang/Downloads/advising_network/reassessed-imagenet/real.json")
            real_ids = json.load(real_json)
            self.real_labels = {
                f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
                for i, labels in enumerate(real_ids)
            }
