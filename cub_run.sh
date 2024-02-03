#!/bin/sh

python cub-200/cub_extract_feature.py  # sampling
python cub-200/augment_images.py
python cub-200/cub_image_comparator_training.py  # train
python cub-200/cub_binary_classify.py  # test