#!/bin/sh

rm -rf /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
cp -r /home/giang/Downloads/Cars/Stanford-Cars-dataset/BACKUP/train /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
CUDA_VISIBLE_DEVICES=4 python cars-196/car_extract_feature.py
python cars-196/augment_images.py
CUDA_VISIBLE_DEVICES=4,5 python cars-196/car_image_comparator_training.py  # train