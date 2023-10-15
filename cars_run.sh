#!/bin/sh

rm -rf /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
cp -r /home/giang/Downloads/Cars/Stanford-Cars-dataset/BACKUP/train /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
python car_extract_feature.py
python augment_images.py
python car_model_training.py
python car_infer.py