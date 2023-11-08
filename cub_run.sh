#!/bin/sh

python cub-200/cub_extract_feature.py
python augment_images.py
python cub-200/cub_model_training.py
python cub-200/cub_infer.py