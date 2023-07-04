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
from PIL import Image
import torchvision.transforms as T

# Define the TrivialAugmentWide transform
trivial_augmenter = T.TrivialAugmentWide()
jitter = T.ColorJitter(brightness=.5, hue=.3)

# Define the RandomApply transform to apply the TrivialAugmentWide transform with a probability of 0.5
trivial_augmenter = T.RandomApply(torch.nn.ModuleList([trivial_augmenter]), p=1.0)

RunningParams = RunningParams()

class ImageFolderForZeroshot(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root, transform=None):
        super(ImageFolderForZeroshot, self).__init__(root, transform=transform)

        self.root = root
        # Load the pre-computed NNs
        if 'test' in os.path.basename(root):
            file_name = 'faiss/NN_dict_NA-Birds.npy'
            # file_name = 'faiss/NN_dict_Dogs.npy'
        else:
            print('Wrong test directory')
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

        nns = self.faiss_nn_dict[base_name]  # a dict of C classes, each class has 6 NNs

        # Initialize an empty tensor to store the transformed images
        tensor_images = torch.empty((len(nns), RunningParams.k_value, 3, 224, 224))

        # Iterate over the dictionary entries and transform the images
        for i, key in enumerate(nns):
            file_paths = nns[key][:RunningParams.k_value]
            for j, file_path in enumerate(file_paths):
                # Load the image using the loader function
                image = self.loader(file_path)  # Replace `loader` with your actual loader function

                # Apply the transformation to the image
                transformed_image = self.transform(image)

                # Assign the transformed image to the tensor
                tensor_images[i, j] = transformed_image

        tuple_with_path = ((query, tensor_images), target, query_path)

        return tuple_with_path

class ImageFolderForAdvisingProcess(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root, transform=None):
        super(ImageFolderForAdvisingProcess, self).__init__(root, transform=transform)

        self.root = root
        # Load the pre-computed NNs
        if 'test' in os.path.basename(root):
            file_name = 'faiss/advising_process_test_top1_HP_MODEL1_HP_FE.npy'
        else:
            file_name = 'faiss/advising_process_val_top1_HP_MODEL1_HP_FE.npy'
            # exit(-1)

        file_name = 'faiss/advising_process_test_Cars.npy'

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

        nns = self.faiss_nn_dict[base_name]  # a dict of C classes, each class has 6 NNs

        # Initialize an empty tensor to store the transformed images
        tensor_images = torch.empty((len(nns), RunningParams.k_value, 3, 224, 224))
        labels = []

        # Iterate over the dictionary entries and transform the images
        for i, val in nns.items():
            file_paths = val['NNs'][:RunningParams.k_value]
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


no_crop_data_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
                            # file_name = 'faiss/cub/top10_k1_enriched_NeurIPS_Finetuning_faiss_train_top1_HP_MODEL1_HP_FE.npy'
                            file_name = '/home/giang/Downloads/advising_network/faiss/cub/NTSNet_10_1_train.npy'

                    else:
                        if RunningParams.MODEL2_FINETUNING is True:
                            if 'val' in os.path.basename(root):
                                # file_name = 'faiss/cub/top1_k1_enriched_NeurIPS_Finetuning_faiss_val_top1_HP_MODEL1_HP_FE.npy'
                                file_name = '/home/giang/Downloads/advising_network/faiss/cub/NTSNet_1_1_val.npy'
                            else:
                                # file_name = 'faiss/cub/top1_k1_enriched_NeurIPS_Finetuning_faiss_test_top1_HP_MODEL1_HP_FE.npy'
                                file_name = '/home/giang/Downloads/advising_network/faiss/cub/NTSNet_1_1_test.npy'

            else:
                if RunningParams.TOP1_NN is True:
                    if 'train' in os.path.basename(root):
                        file_name = 'faiss/dogs/faiss_SDogs_RN34_SDogs_train_RN34_top5_{}NNs.npy'.format(RunningParams.k_value)
                        file_name = '/home/giang/Downloads/advising_network/faiss/cars/top10_k1_enriched_NeurIPS_Finetuning_faiss_train_top1.npy'
                    elif 'val' in os.path.basename(root):
                        # file_name = 'faiss/dogs/faiss_SDogs_RN34_SDogs_val_RN34_top1_{}NNs.npy'.format(RunningParams.k_value)
                        file_name = '/home/giang/Downloads/advising_network/faiss/cars/top1_k1_enriched_NeurIPS_Finetuning_faiss_test_top1.npy'
                    elif 'test' in os.path.basename(root):
                        file_name = '/home/giang/Downloads/advising_network/faiss/cars/top1_k1_enriched_NeurIPS_Finetuning_faiss_test_top1.npy'
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
            if 'train' in os.path.basename(self.root):
                sample = trivial_augmenter(sample)

            sample = self.transform(sample)
            explanations.append(sample)
        # If query is the same with any of NNs --> duplicate the last element
        if dup is True:
            explanations += [explanations[-1]]
        explanations = torch.stack(explanations)

        # Transform query
        sample = self.loader(query_path)
        query = self.transform(sample)

        no_crop = trivial_augmenter(sample)
        no_crop = self.transform(no_crop)

        # make a new tuple that includes original and the path
        if 'train' in os.path.basename(self.root):
            tuple_with_path = ((query, explanations, model2_target, no_crop), target, query_path)
        else:
            tuple_with_path = ((query, explanations, no_crop), target, query_path)

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

import shutil
from scipy.io import loadmat
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
        # if not os.path.isdir(os.path.join(root, "images")):
        # self.download()
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

    def download(self):
        """Download the dataset"""
        downloads_dir = os.path.join(self.root, "downloads")
        data_dir = os.path.join(self.root, "images")
        try:
            pass
        except FileNotFoundError:
            pass
        finally:
            # os.mkdir(self.root)
            # os.mkdir(downloads_dir)
            pass

        # Split images into train, validation, and test sets
        print("Splitting dataset")
        # os.rmdir(data_dir)
        # os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "train"))
        os.mkdir(os.path.join(data_dir, "validation"))
        train_list = [f[0][0] for f in loadmat(os.path.join(downloads_dir, "train_list.mat"))["file_list"]]
        # Shuffle the training images
        random.shuffle(train_list)
        for (i, file) in enumerate(train_list):
            if i < 200:
                # The first 200 training images get put into the validation directory
                target_dir = os.path.join(data_dir, "validation")
            else:
                # The rest go into the train directory
                target_dir = os.path.join(data_dir, "train")
            try:
                # Create the directory for the breed if it doesn't exist
                os.mkdir(os.path.join(target_dir, os.path.split(file)[0]))
            except FileExistsError:
                # The directory was already there
                pass
            finally:
                # Move the image
                shutil.move(os.path.join(downloads_dir, "Images", file), os.path.join(target_dir, file))
        # Move the test images
        os.mkdir(os.path.join(data_dir, "test"))
        test_list = loadmat(os.path.join(downloads_dir, "test_list.mat"))["file_list"]
        for file in test_list:
            if not os.path.isdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0])):
                # Create the directory for the breed if it doesn't exist
                os.mkdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0]))
            # Move the image
            shutil.move(os.path.join(downloads_dir, "Images", file[0][0]), os.path.join(data_dir, "test", file[0][0]))
        # shutil.move(os.path.join(downloads_dir, "file_list.mat"))
        # shutil.rmtree(downloads_dir)
        print("Splitting complete")

    def get_labels(self):
        subdirs = set()
        labels = {}
        for subdir, _, _ in os.walk(os.path.join(self.root, "images/test")):
            if (label := os.path.split(subdir)[-1]) != "test":
                subdirs |= {label}
                labels[label] = len(subdirs) - 1
        return labels