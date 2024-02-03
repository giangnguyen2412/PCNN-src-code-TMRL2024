#!/bin/sh

python stanford-dogs/car_extract_feature.py
python stanford-dogs/augment_images.py
python stanford-dogs/car_image_comparator_training.py  # train
python stanford-dogs/car_binary_classify.py  # test