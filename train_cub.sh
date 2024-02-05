#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python cub-200/cub_extract_feature.py  # sampling
python cub-200/augment_images.py
CUDA_VISIBLE_DEVICES=4,5 python cub-200/cub_image_comparator_training.py  # train
