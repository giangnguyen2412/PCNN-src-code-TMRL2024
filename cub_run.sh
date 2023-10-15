#!/bin/sh

python cub_extract_feature.py
python augment_images.py
python cub_model_training.py
python cub_infer.py