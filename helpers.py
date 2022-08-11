import psutil
import os
import shutil
from shutil import *

import numpy as np
import torch
from PIL import Image


class HelperFunctions(object):
    def __init__(self):
        # Some Helper functions
        self.id_map = self.load_imagenet_id_map()
        self.label_map = self.load_imagenet_label_map()
        self.key_list = list(self.id_map.keys())
        self.val_list = list(self.id_map.values())

    def concat(self, x):
        return np.concatenate(x, axis=0)

    def to_np(self, x):
        return x.data.to("cpu").numpy()

    def to_ts(self, x):
        return torch.from_numpy(x)

    def train_extract_wnid(self, x):
        return x.split("train/")[1].split("/")[0]

    def val_extract_wnid(self, x):
        # return x.split(x + '/')[1].split('/')[0]
        return x.split("/")[-2]

    def rm_and_mkdir(self, path):
        if os.path.isdir(path) == True:
            rmtree(path)
        os.mkdir(path)

    def copy_files(self, files, dir):
        for file in files:
            shutil.copy(file, dir)

    def is_grey_scale(self, img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    def check_and_mkdir(self, f):
        if not os.path.exists(f):
            os.mkdir(f)
        else:
            pass

    def check_and_rm(self, f):
        if os.path.exists(f):
            shutil.rmtree(f)
        else:
            pass

    def load_imagenet_label_map(self):
        """
        Load ImageNet label dictionary.
        return:
        """

        input_f = open("/home/giang/Downloads/kNN-classifiers/input_txt_files/imagenet_classes.txt")
        label_map = {}
        for line in input_f:
            parts = line.strip().split(": ")
            (num, label) = (int(parts[0]), parts[1].replace('"', ""))
            label_map[num] = label

        input_f.close()
        return label_map

    # Added for loading ImageNet classes
    def load_imagenet_id_map(self):
        """
        Load ImageNet ID dictionary.
        return;
        """

        input_f = open("/home/giang/Downloads/KNN-ImageNet/synset_words.txt")
        label_map = {}
        for line in input_f:
            parts = line.strip().split(" ")
            (num, label) = (parts[0], " ".join(parts[1:]))
            label_map[num] = label

        input_f.close()
        return label_map

    def convert_imagenet_label_to_id(
        self, label_map, key_list, val_list, prediction_class
    ):
        """
        Convert imagenet label to ID: for example - 245 -> "French bulldog" -> n02108915
        :param label_map:
        :param key_list:
        :param val_list:
        :param prediction_class:
        :return:
        """
        class_to_label = label_map[prediction_class]
        prediction_id = key_list[val_list.index(class_to_label)]
        return prediction_id

    def convert_imagenet_id_to_label(self, key_list, class_id):
        """
        Convert imagenet label to ID: for example - n02108915 -> "French bulldog" -> 245
        :param label_map:
        :param key_list:
        :param val_list:
        :param prediction_class:
        :return:
        """
        return key_list.index(str(class_id))

    @staticmethod
    def is_program_running(script):
        """
        Check if a script is already running
        :param script:
        :return:
        """
        for q in psutil.process_iter():
            if q.name().startswith('python'):
                if len(q.cmdline())>1 and script in q.cmdline()[1] and q.pid !=os.getpid():
                    print("'{}' Process is already running".format(script))
                    return True

        return False