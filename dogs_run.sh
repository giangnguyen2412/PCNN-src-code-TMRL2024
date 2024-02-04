#!/bin/sh

python stanford-dogs/dogs_extract_feature.py
python stanford-dogs/augment_images.py
python stanford-dogs/dogs_image_comparator_training.py  # train
python stanford-dogs/dogs_binary_classify.py  # test