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
import traceback
from PIL import Image
import torchvision.transforms as T

# Define the TrivialAugmentWide transform
trivial_augmenter = T.TrivialAugmentWide()

# Define the RandomApply transform to apply the TrivialAugmentWide transform with a probability of 0.5
trivial_augmenter = T.RandomApply(torch.nn.ModuleList([trivial_augmenter]), p=0.5)

RunningParams = RunningParams()

class ImageFolderForAdvisingProcess(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root, transform=None, nn_num=None):
        super(ImageFolderForAdvisingProcess, self).__init__(root, transform=transform)

        self.root = root
        # Load the pre-computed NNs
        if RunningParams.CUB_TRAINING is True:
            if 'test' in os.path.basename(root):
                file_name = f'{RunningParams.prj_dir}/faiss/advising_process_test_top1_HP_MODEL1_HP_FE.npy'
            else:
                file_name = f'{RunningParams.prj_dir}/faiss/advising_process_val_top1_HP_MODEL1_HP_FE.npy'

        elif RunningParams.CARS_TRAINING is True:
            file_name = f'{RunningParams.prj_dir}/faiss/advising_process_test_Cars.npy'

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
        self.nn_num = nn_num

    def __getitem__(self, index):
        query_path, target = self.samples[index]

        # Transform query
        sample = self.loader(query_path)
        query = self.transform(sample)

        base_name = os.path.basename(query_path)

        nns = self.faiss_nn_dict[base_name]  # a dict of C classes, each class has 6 NNs

        if self.nn_num is None:
            nn_num = RunningParams.k_value
        else:
            nn_num = self.nn_num

        # Initialize an empty tensor to store the transformed images
        tensor_images = torch.empty((len(nns), nn_num, 3, 224, 224))

        labels = []

        # Iterate over the dictionary entries and transform the images
        for i, val in nns.items():
            # Only take either the first or second, or third NNs for the comparison
            file_paths = val['NNs'][RunningParams.negative_order-1:RunningParams.negative_order]
            labels.append(val['Label'])
            for j, file_path in enumerate(file_paths):
                # Load the image using the loader function
                image = self.loader(file_path)  # Replace `loader` with your actual loader function

                # Apply the transformation to the image
                transformed_image = self.transform(image)

                # Assign the transformed image to the tensor
                tensor_images[i, j] = transformed_image

        labels = torch.tensor(labels)
        tuple_with_path = ((query, tensor_images, labels), target, query_path)

        return tuple_with_path

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __init__ method to drop no-label images
    def __init__(self, root, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform=transform)

        original_len = len(self.imgs)
        imgs = []
        samples = []
        targets = []
        for sample_idx in range(original_len):
            pth = self.imgs[sample_idx][0]
            base_name = os.path.basename(pth)

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

    def __init__(self, root, transform=None, nn_dict=None):
        super(ImageFolderForNNs, self).__init__(root, transform=transform)

        self.root = root
        # Load the pre-computed NNs
        if RunningParams.CUB_TRAINING is True:
            if 'train' in os.path.basename(root):
                file_name = RunningParams.faiss_npy_file
            elif 'test' in os.path.basename(root):
                # file_name = 'faiss/cub/top1_k1_enriched_NeurIPS_Finetuning_faiss_test_top1_HP_MODEL1_HP_FE.npy'
                file_name = RunningParams.faiss_npy_file
            else:
                file_name = RunningParams.faiss_npy_file

        elif RunningParams.CARS_TRAINING is True:
            if 'train' in os.path.basename(root):
                file_name = RunningParams.faiss_npy_file
            elif 'test' in os.path.basename(root):
                file_name = 'faiss/cars/top1_k1_enriched_NeurIPS_Finetuning_faiss_test_top1.npy'
            else:
                file_name = RunningParams.faiss_npy_file

        else:
            print("Wrong RunningParams params!")
            exit(-1)

        if nn_dict is not None:
            file_name = nn_dict

        print(file_name)
        self.faiss_nn_dict = np.load(file_name, allow_pickle=True, ).item()

        sample_count = len(self.faiss_nn_dict)
        print(sample_count)


    def __getitem__(self, index):
        query_path, target = self.samples[index]
        base_name = os.path.basename(query_path)
        if RunningParams.XAI_method == RunningParams.NNs:
            if 'train' in os.path.basename(self.root):

                nns = self.faiss_nn_dict[base_name]['NNs']  # 6NNs here
                model2_target = self.faiss_nn_dict[base_name]['label']
            elif 'val' in os.path.basename(self.root) and (RunningParams.CARS_TRAINING is True or
                                                         RunningParams.DOGS_TRAINING is True):
                nns = self.faiss_nn_dict[base_name]['NNs']  # 6NNs here
                model2_target = self.faiss_nn_dict[base_name]['label']
            else:
                # nns = self.faiss_nn_dict[base_name]  # 6NNs here
                nns = self.faiss_nn_dict[base_name]['NNs']  # 6NNs here
                model2_target = self.faiss_nn_dict[base_name]['label']

        # Transform NNs
        explanations = list()
        dup = False
        for pth in nns:
            sample = self.loader(pth)
            nn_base_name = os.path.basename(pth)
            if nn_base_name in base_name:
                dup = True
                continue
            if 'train' in os.path.basename(self.root):
                sample = trivial_augmenter(sample)

            sample = self.transform(sample)
            explanations.append(sample)
        # If query is the same with any of NNs --> wrongly retrieved NNs
        if dup is True:
            print('I found the query and the NNs are the same file. Duplicate detected! Exit now ...')
            exit(-1)

        explanations = torch.stack(explanations)

        # Transform query
        sample = self.loader(query_path)
        query = self.transform(sample)

        aug_query = trivial_augmenter(sample)
        aug_query = self.transform(aug_query)

        # make a new tuple that includes original and the path
        if 'train' in os.path.basename(self.root):
                tuple_with_path = ((query, explanations, model2_target, aug_query), target, query_path)
        elif 'val' in os.path.basename(self.root) and (RunningParams.CARS_TRAINING is True or
                                                       RunningParams.DOGS_TRAINING is True):
            tuple_with_path = ((query, explanations, model2_target, aug_query), target, query_path)
        else:
            tuple_with_path = ((query, explanations, aug_query), target, query_path)

        return tuple_with_path


class Dataset(object):
    def __init__(self):
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


class StanfordDogsDataset(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

    Args:
        root (string): Directory where the data is stored
        set_type (string, optional): Specify `train`, `validation`, or `test`. If
            unspecified, it is taken as `test`.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed tensor.
    """

    def __init__(self, root, set_type="test", transform=T.ToTensor):
        self.root = root
        self.transform = transform
        self.file_paths = []
        self.labels = []
        label_names = self.get_labels()

        self.mapping = dict()
        for k, v in label_names.items():
            self.mapping[v] = k

        for dirpath, _, files in os.walk(os.path.join(root, "images", set_type)):
            for file in files:
                self.file_paths.append(os.path.join(dirpath, file))
                self.labels.append(label_names[os.path.split(dirpath)[-1]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.file_paths[item])
        image = self.transform(image)
        image = torch.from_numpy(np.asarray(image))

        return image, torch.tensor(self.labels[item])

    def get_labels(self):
        subdirs = set()
        labels = {}
        for subdir, _, _ in os.walk(os.path.join(self.root, "images/test")):
            if (label := os.path.split(subdir)[-1]) != "test":
                subdirs |= {label}
                labels[label] = len(subdirs) - 1
        return labels