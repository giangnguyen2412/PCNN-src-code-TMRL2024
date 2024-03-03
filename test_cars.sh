#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python cars-196/car_extract_feature.py  # sampling for test set
python cars-196/clean_dups.py
python cars-196/augment_images.py
CUDA_VISIBLE_DEVICES=4 python cars-196/car_binary_classify.py  # binary classification

CUDA_VISIBLE_DEVICES=4 python cars-196/car_extract_feature_for_reranking.py  # sampling for reranking
CUDA_VISIBLE_DEVICES=4,5 python cars-196/car_reranking.py  # reranking