#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python stanford-dogs/dogs_extract_feature.py
python stanford-dogs/augment_images.py
CUDA_VISIBLE_DEVICES=4,5 python stanford-dogs/dogs_image_comparator_training.py  # train
