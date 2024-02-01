#!/bin/sh

rm -rf /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
cp -r /home/giang/Downloads/Cars/Stanford-Cars-dataset/BACKUP/train /home/giang/Downloads/Cars/Stanford-Cars-dataset/train/
python cars-196/car_extract_feature.py
python augment_images.py
python cars-196/car_model_training.py
python cars-196/car_infer.py