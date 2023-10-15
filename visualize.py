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
from matplotlib.colors import ListedColormap
from params import *
from datasets import Dataset, ImageFolderWithPaths
from torchvision import datasets, models, transforms

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
        # from PIL import Image
        #
        # # Load and resize the query image
        # query_img = Image.open(query)
        # query_img = query_img.resize((400, 400))
        #
        # # Create a new figure
        # fig, ax = plt.subplots()
        # ax.imshow(query_img)
        # ax.set_title(f'Query: {gt_label}', fontsize=16, color='green',
        #              weight='bold')
        # ax.axis('off')
        # plt.savefig("query.jpg", bbox_inches='tight', pad_inches=0.15)
        #
        # # Load and resize the query image
        # prototype_img = Image.open(prototypes[0])
        # prototype_img = prototype_img.resize((400, 400))
        #
        # # Create a new figure
        # fig, ax = plt.subplots()
        # ax.imshow(prototype_img)
        # if gt_label == pred_label:
        #     color = 'green'
        # else:
        #     color = 'red'
        # ax.set_title(f'C\'s top-1: {pred_label}', fontsize=16, color=color,
        #              weight='bold')
        #
        # color = 'red'
        # # Add the confidence at the bottom of the image
        # ax.text(0.5, -0.07, f'AdvNet\'s Confidence: {confidence2:.2f}%', size=12, ha="center",
        #                 transform=ax.transAxes, weight='bold', color=color)
        #
        # ax.axis('off')
        # plt.savefig("prototype.jpg", bbox_inches='tight', pad_inches=0.15)
        #
        # cmd = 'montage query.jpg prototype.jpg -tile 2x1 -geometry +2+2 {}'.format(save_path)
        # os.system(cmd)

        from PIL import Image

        # Load and resize the query image
        query_img = Image.open(query)
        query_img = query_img.resize((400, 400))

        # Load and resize the prototype image
        prototype_img = Image.open(prototypes[0])
        prototype_img = prototype_img.resize((400, 400))

        # Create a new figure with one row and two columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Plot the query image in the first column
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query: {gt_label}', fontsize=16, color='green', weight='bold')
        axes[0].axis('off')

        # Plot the prototype image in the second column
        axes[1].imshow(prototype_img)
        if gt_label == pred_label:
            color = 'green'
        else:
            color = 'red'
        axes[1].set_title(f"C's top-1: {pred_label}", fontsize=16, color=color, weight='bold')

        color = 'red'
        # Add the confidence at the bottom of the image
        axes[1].text(0.5, -0.07, f"AdvNet's Confidence: {confidence2:.2f}%", size=24, ha="center",
                     transform=axes[1].transAxes, weight='bold', color=color)

        axes[1].axis('off')

        # Save the figure as a PDF
        plt.tight_layout()
        save_path = save_path.replace('jpg', 'pdf')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.25)

    @staticmethod
    def visualize_model2_correction_with_prototypes(query: str,
                                                    gt_label: str,
                                                    pred_label: str,
                                                    model2_decision: str,
                                                    adv_label: str,
                                                    save_path: str,
                                                    save_dir: str,
                                                    confidence1: int,
                                                    confidence2: int,
                                                    adv_conf: int,
                                                    prototypes: list,
                                                    adv_prototypes: list
                                                    ):
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            query, save_dir
        )
        os.system(cmd)
        for idx, prototype in enumerate(prototypes):
            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                prototype, save_dir, idx)
            os.system(cmd)

        annotation = 'GT: {} - Model1 Pred: {} - Model1 Conf: {}% - Model2 Conf: {}%'.format(
            gt_label, pred_label, confidence1, confidence2)

        cmd = 'montage {}/[0-4].jpeg -tile 3x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
        os.system(cmd)
        cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}'.format(save_dir, save_dir,
                                                                                           save_path)
        os.system(cmd)

        cmd = 'convert {} -font aakar -pointsize 16 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}'.format(
            save_path, annotation, 'tmp1.jpg'
        )
        os.system(cmd)

        ################################################################################

        for idx, prototype in enumerate(adv_prototypes):
            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                prototype, save_dir, idx)
            os.system(cmd)

        annotation = 'GT: {} - Model1 Pred: {} - Model1 Conf: < {}% - Model2 Conf: {}%'.format(
            gt_label, adv_label, 100-confidence1, adv_conf)

        cmd = 'montage {}/[0-4].jpeg -tile 3x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
        os.system(cmd)
        cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}'.format(save_dir, save_dir,
                                                                                           save_path)
        os.system(cmd)

        cmd = 'convert {} -font aakar -pointsize 16 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}'.format(
            save_path, annotation, 'tmp2.jpg'
        )
        os.system(cmd)

        cmd = 'montage tmp1.jpg tmp2.jpg -tile x2 -geometry +2+2 {}'.format(save_path)
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
    #         query = os.path.join(f'{RunningParams.parent_dir}/datasets/Dogs_train', wnid, query)
    #     else:
    #         path = glob.glob(f'{RunningParams.parent_dir}/datasets/imagenet1k-val/**/{}'.format(query))
    #         wnid = path[0].split('/')[-2]
    #         query = os.path.join(f'{RunningParams.parent_dir}/datasets/Dogs_val', wnid, query)
    #
    #     gt_label = HelperFunctions.convert_imagenet_id_to_label(HelperFunctions.key_list, wnid)
    #     gt_label = HelperFunctions.label_map[gt_label]
    #     visualize_dog_prototypes(query, gt_label, nn, 'tmp')

    @staticmethod
    def visualize_transformer_attn(bef_weights, aft_weights, bef_image_paths, aft_image_paths, title, proto_idx):

        input_label = os.path.dirname(aft_image_paths).split('/')[-1]
        prototype_label = os.path.dirname(bef_image_paths).split('/')[-1]
        basename = os.path.basename(aft_image_paths)

        from skimage import exposure
        from PIL import Image
        def load_image(path):
            image = Image.open(path)
            image = image.resize((224, 224), Image.ANTIALIAS)
            image = np.array(image)
            return image

        bef_image_path = bef_image_paths
        aft_image_path = aft_image_paths

        bef_image = load_image(bef_image_path)
        aft_image = load_image(aft_image_path)

        bef_weights = bef_weights.data.cpu().numpy().reshape(7, 7)
        aft_weights = aft_weights.data.cpu().numpy().reshape(7, 7)

        cam_bef_weights = exposure.rescale_intensity(bef_weights, oudt_range=(0, 255))
        cam_bef_weights = cv2.resize(cam_bef_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

        cam_aft_weights = exposure.rescale_intensity(aft_weights, out_range=(0, 255))
        cam_aft_weights = cv2.resize(cam_aft_weights.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(bef_image)
        ax1.axis("off")
        ax1.set_title(prototype_label)

        ax2.imshow(aft_image)
        ax2.axis("off")
        ax2.set_title(input_label)

        ax3.imshow(bef_image)
        ax3.imshow(cam_bef_weights, alpha=0.6, cmap="jet")
        ax3.axis("off")

        ax4.imshow(aft_image)
        ax4.imshow(cam_aft_weights, alpha=0.6, cmap="jet")
        ax4.axis("off")

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.2)
        if proto_idx != 1:
            plt.suptitle('', fontsize=16, y=0.02)
        else:
            plt.suptitle(title, fontsize=16, y=0.02)

        HelperFunctions.check_and_mkdir('tmp/{}'.format(title))
        plt.savefig('tmp/{}/{}_{}.png'.format(title, basename, proto_idx), bbox_inches='tight')

    @staticmethod
    def visualize_transformer_attn_v2(bef_weights, aft_weights, bef_image_paths, aft_image_paths, title, proto_idx):

        input_label = os.path.dirname(aft_image_paths).split('/')[-1]
        prototype_label = os.path.dirname(bef_image_paths).split('/')[-1]
        basename = os.path.basename(aft_image_paths)

        from skimage import exposure
        from PIL import Image
        def load_image(path):
            image = Image.open(path)
            image = image.resize((224, 224), Image.ANTIALIAS)
            image = np.array(image)
            return image

        bef_image_path = bef_image_paths
        aft_image_path = aft_image_paths

        bef_image = load_image(bef_image_path)
        aft_image = load_image(aft_image_path)

        bef_weights = bef_weights.data.cpu().numpy().reshape(7, 7)
        aft_weights = aft_weights.data.cpu().numpy().reshape(7, 7)

        cam_bef_weights = bef_weights
        # Normalize the array to a range of 0 and 1
        cam_bef_weights = (cam_bef_weights - cam_bef_weights.min()) / (cam_bef_weights.max() - cam_bef_weights.min())
        cam_bef_weights = cv2.resize(cam_bef_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)

        cam_aft_weights = aft_weights
        cam_aft_weights = (cam_aft_weights - cam_aft_weights.min()) / (cam_aft_weights.max() - cam_aft_weights.min())
        cam_aft_weights = cv2.resize(cam_aft_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)

        uP = cm.get_cmap('Reds', 129)
        dowN = cm.get_cmap('Blues_r', 128)

        newcolors = np.vstack((
            dowN(np.linspace(0, 1, 128)),
            uP(np.linspace(0, 1, 129))
        ))
        cMap = ListedColormap(newcolors, name='RedBlues')

        cMap.colors[257 // 2, :] = [1, 1, 1, 1]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(bef_image)
        ax1.axis("off")
        ax1.set_title(prototype_label)

        ax2.imshow(aft_image)
        ax2.axis("off")
        ax2.set_title(input_label)

        ax3.imshow(bef_image)
        im1 = ax3.imshow(cam_bef_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax3.axis("off")

        ax4.imshow(aft_image)
        im2 = ax4.imshow(cam_aft_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax4.axis("off")

        fig.subplots_adjust(right=0.8)
        # if proto_idx != 1:
        #     plt.suptitle('', fontsize=16, y=0.02)
        # else:
        plt.suptitle(title, fontsize=16, y=0.02)
        fig.colorbar(im2, ax=ax4)
        fig.colorbar(im1, ax=ax3)

        HelperFunctions.check_and_mkdir('tmp/{}'.format(title))
        plt.savefig('tmp/{}/{}_{}.png'.format(title, basename, proto_idx), bbox_inches='tight')

    @staticmethod
    def visualize_transformer_attn_birds(bef_weights, aft_weights, bef_image_paths, aft_image_paths, title, proto_idx):

        input_label = os.path.dirname(aft_image_paths).split('/')[-1]
        prototype_label = os.path.dirname(bef_image_paths).split('/')[-1]
        basename = os.path.basename(aft_image_paths)

        from PIL import Image
        from PIL import Image

        def load_image(path):
            image = Image.open(path)

            # Step 1: Determine aspect ratio
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:  # width > height
                new_width = int(256 * aspect_ratio)
                new_height = 256
            else:
                new_width = 256
                new_height = int(256 / aspect_ratio)

            # Step 2: Resize the image
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Step 3: Compute the coordinates for a 224x224 center crop
            left = (new_width - 224) / 2
            top = (new_height - 224) / 2
            right = (new_width + 224) / 2
            bottom = (new_height + 224) / 2

            # Step 4: Perform the crop
            image = image.crop((left, top, right, bottom))

            image = np.array(image)
            return image

        bef_image_path = bef_image_paths
        aft_image_path = aft_image_paths

        bef_image = load_image(bef_image_path)
        aft_image = load_image(aft_image_path)

        bef_weights = bef_weights.data.cpu().numpy().reshape(7, 7)
        aft_weights = aft_weights.data.cpu().numpy().reshape(7, 7)

        cam_bef_weights = bef_weights
        # Normalize the array to a range of 0 and 1
        cam_bef_weights = (cam_bef_weights - cam_bef_weights.min()) / (cam_bef_weights.max() - cam_bef_weights.min())
        cam_bef_weights = cv2.resize(cam_bef_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)

        cam_aft_weights = aft_weights
        cam_aft_weights = (cam_aft_weights - cam_aft_weights.min()) / (cam_aft_weights.max() - cam_aft_weights.min())
        cam_aft_weights = cv2.resize(cam_aft_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)

        uP = cm.get_cmap('Reds', 129)
        dowN = cm.get_cmap('Blues_r', 128)

        newcolors = np.vstack((
            dowN(np.linspace(0, 1, 128)),
            uP(np.linspace(0, 1, 129))
        ))
        cMap = ListedColormap(newcolors, name='RedBlues')

        cMap.colors[257 // 2, :] = [1, 1, 1, 1]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(bef_image)
        ax1.axis("off")
        ax1.set_title(prototype_label)

        ax2.imshow(aft_image)
        ax2.axis("off")
        ax2.set_title(input_label)

        ax3.imshow(bef_image)
        im1 = ax3.imshow(cam_bef_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax3.axis("off")

        ax4.imshow(aft_image)
        im2 = ax4.imshow(cam_aft_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax4.axis("off")

        fig.subplots_adjust(right=0.8)
        # if proto_idx != 1:
        #     plt.suptitle('', fontsize=16, y=0.02)
        # else:
        # plt.suptitle(title, fontsize=16, y=0.02)
        fig.colorbar(im2, ax=ax4)
        fig.colorbar(im1, ax=ax3)

        HelperFunctions.check_and_mkdir('attn_maps/cub')
        HelperFunctions.check_and_mkdir('attn_maps/cub/{}'.format(title))
        plt.savefig('attn_maps/cub/{}/{}.pdf'.format(title, basename), bbox_inches='tight', dpi=300)

    @staticmethod
    def visualize_transformer_attn_sdogs(bef_weights, aft_weights, bef_image_paths, aft_image_paths, title, proto_idx):
        breakpoint()
        input_label = os.path.dirname(aft_image_paths).split('/')[-1]
        prototype_label = os.path.dirname(bef_image_paths).split('/')[-1]
        basename = os.path.basename(aft_image_paths)

        from skimage import exposure
        from PIL import Image
        def load_image(path):
            image = Image.open(path)
            image = image.resize((512, 512), Image.ANTIALIAS)
            image = np.array(image)
            return image

        bef_image_path = bef_image_paths
        aft_image_path = aft_image_paths

        bef_image = load_image(bef_image_path)
        aft_image = load_image(aft_image_path)

        bef_weights = bef_weights.data.cpu().numpy().reshape(16, 16)
        aft_weights = aft_weights.data.cpu().numpy().reshape(16, 16)

        cam_bef_weights = bef_weights
        # Normalize the array to a range of 0 and 1
        cam_bef_weights = (cam_bef_weights - cam_bef_weights.min()) / (
                    cam_bef_weights.max() - cam_bef_weights.min())
        cam_bef_weights = cv2.resize(cam_bef_weights.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)

        cam_aft_weights = aft_weights
        cam_aft_weights = (cam_aft_weights - cam_aft_weights.min()) / (
                    cam_aft_weights.max() - cam_aft_weights.min())
        cam_aft_weights = cv2.resize(cam_aft_weights.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)

        uP = cm.get_cmap('Reds', 129)
        dowN = cm.get_cmap('Blues_r', 128)

        newcolors = np.vstack((
            dowN(np.linspace(0, 1, 128)),
            uP(np.linspace(0, 1, 129))
        ))
        cMap = ListedColormap(newcolors, name='RedBlues')

        cMap.colors[257 // 2, :] = [1, 1, 1, 1]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.imshow(bef_image)
        ax1.axis("off")
        ax1.set_title(prototype_label)

        ax2.imshow(aft_image)
        ax2.axis("off")
        ax2.set_title(input_label)

        ax3.imshow(bef_image)
        im1 = ax3.imshow(cam_bef_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax3.axis("off")

        ax4.imshow(aft_image)
        im2 = ax4.imshow(cam_aft_weights, alpha=0.5, cmap='Reds', interpolation='none', vmin=0, vmax=1)
        ax4.axis("off")

        fig.subplots_adjust(right=0.8)
        plt.suptitle(title, fontsize=16, y=0.02)
        fig.colorbar(im2, ax=ax4)
        fig.colorbar(im1, ax=ax3)

        HelperFunctions.check_and_mkdir('attn_maps/{}'.format(title))
        plt.savefig('attn_maps/{}/{}_{}.jpg'.format(title, basename, proto_idx), bbox_inches='tight')

