import os.path

import cv2
import cv2 as cv
import image_slicer
import matplotlib
import matplotlib.patches as patches
import torch.nn.functional as F
import seaborn as sns
from helpers import *
from image_slicer import join
from IPython.display import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from params import *
from datasets import Dataset, ImageFolderWithPaths

RunningParams = RunningParams()
HelperFunctions = HelperFunctions()

sns.set(style="darkgrid")


class Visualization(object):
    def __init__(self):
        uP = cm.get_cmap("turbo", 96)
        self.cMap = uP

        # The images to AIs should be the same with the ones to humans
        self.display_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        cmap = matplotlib.cm.get_cmap("gist_rainbow")
        self.colors = []
        for k in range(5):
            self.colors.append(cmap(k / 5.0))

        self.upsample_size = 224

    @staticmethod
    def visualize_histogram_from_list(data: list,
                                      title: str,
                                      x_label: str,
                                      y_label: str,
                                      file_name: str):
        sns.histplot(data=data, kde=True, bins=20)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def visualize_model2_decisions(query: str,
                                   gt_label: str,
                                   pred_label: str,
                                   model2_decision: str,
                                   save_path: str,
                                   save_dir: str,
                                   confidence: int
                                   ):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            query, save_dir
        )
        os.system(cmd)

        annotation = 'GT: {} > Pred: {} > Conf: {}% > Model2: {}'.format(gt_label, pred_label, confidence,
                                                                         model2_decision)

        cmd = 'convert {}/query.jpeg -resize 600x600\! -pointsize 14 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}'.format(
            save_dir, annotation, save_path
        )
        os.system(cmd)

    @staticmethod
    def visualize_model2_decision_with_prototypes(query: str,
                                   gt_label: str,
                                   pred_label: str,
                                   model2_decision: str,
                                   save_path: str,
                                   save_dir: str,
                                   confidence1: int,
                                   confidence2: int,
                                   prototypes: list,
                                   ):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            query, save_dir
        )
        os.system(cmd)
        for idx, prototype in enumerate(prototypes):
            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                prototype, save_dir, idx)
            os.system(cmd)

        annotation = 'GT: {} - Model1 Pred: {} - Model1 Conf: {}% - Model2 Conf: {}% - Category: {}'.format(
            gt_label, pred_label, confidence1, confidence2, model2_decision)

        cmd = 'montage {}/[0-4].jpeg -tile 5x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
        os.system(cmd)
        cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}'.format(save_dir, save_dir, save_path)
        os.system(cmd)

        cmd = 'convert {} -font aakar -pointsize 21 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}'.format(
            save_path, annotation, save_path
        )
        print(cmd)
        os.system(cmd)

    @staticmethod
    def visualize_cub200_prototypes(
            query: str,
            gt_label: str,
            prototypes: list,
            save_dir: str):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            query, save_dir
        )
        os.system(cmd)
        annotation = gt_label
        cmd = 'convert {}/query.jpeg -font aakar -pointsize 10 -gravity North -background ' \
              'White -splice 0x40 -annotate +0+4 "{}" {}/query.jpeg'.format(
            save_dir, annotation, save_dir
        )
        os.system(cmd)
        for idx, prototype in enumerate(prototypes):
            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                prototype, save_dir, idx)
            os.system(cmd)
            annotation = prototype.split('/')[-2]
            cmd = 'convert {}/{}.jpeg -font aakar -pointsize 10 -gravity North -background ' \
                  'White -splice 0x40 -annotate +0+4 "{}" {}/{}.jpeg'.format(
                save_dir, idx, annotation, save_dir, idx
            )
            os.system(cmd)

        cmd = 'montage {}/[0-5].jpeg -tile 6x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
        os.system(cmd)
        cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}/{}.JPEG'.format(save_dir, save_dir,
                                                                                                   save_dir, gt_label)
        os.system(cmd)

    @staticmethod
    # filename = 'faiss/faiss_SDogs_val_RN34_top1.npy'
    # kbc = np.load(filename, allow_pickle=True, ).item()
    #
    # HelperFunctions = HelperFunctions()
    def visualize_dog_prototypes(
            query: str,
            gt_label: str,
            prototypes: list,
            save_dir: str):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            query, save_dir
        )
        os.system(cmd)
        annotation = gt_label
        cmd = 'convert {}/query.jpeg -font aakar -pointsize 10 -gravity North -background ' \
              'White -splice 0x40 -annotate +0+4 "{}" {}/query.jpeg'.format(
            save_dir, annotation, save_dir
        )
        os.system(cmd)
        for idx, prototype in enumerate(prototypes):
            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                prototype, save_dir, idx)
            os.system(cmd)
            annotation = prototype.split('/')[-2]

            annotation = HelperFunctions.convert_imagenet_id_to_label(HelperFunctions.key_list, annotation)
            annotation = HelperFunctions.label_map[annotation]

            cmd = 'convert {}/{}.jpeg -font aakar -pointsize 10 -gravity North -background ' \
                  'White -splice 0x40 -annotate +0+4 "{}" {}/{}.jpeg'.format(
                save_dir, idx, annotation, save_dir, idx
            )
            os.system(cmd)

        cmd = 'montage {}/[0-5].jpeg -tile 6x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
        os.system(cmd)

        file_name = os.path.basename(query)
        cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}/{}.JPEG'.format(save_dir, save_dir,
                                                                                                   save_dir, file_name)
        os.system(cmd)

    # id = HelperFunctions.load_imagenet_validation_gt()
    # cnt = 0
    #
    # for query, nn in kbc.items():
    #     cnt += 1
    #     if cnt == 10:
    #         break
    #
    #     if 'train' in filename:
    #         wnid = query.split('_')[0]
    #         query = os.path.join('/home/giang/Downloads/datasets/Dogs_train', wnid, query)
    #     else:
    #         path = glob.glob('/home/giang/Downloads/datasets/imagenet1k-val/**/{}'.format(query))
    #         wnid = path[0].split('/')[-2]
    #         query = os.path.join('/home/giang/Downloads/datasets/Dogs_val', wnid, query)
    #
    #     gt_label = HelperFunctions.convert_imagenet_id_to_label(HelperFunctions.key_list, wnid)
    #     gt_label = HelperFunctions.label_map[gt_label]
    #     visualize_dog_prototypes(query, gt_label, nn, 'tmp')



