#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python cub-200/cub_extract_feature.py  # sampling for test set
python cub-200/augment_images.py
CUDA_VISIBLE_DEVICES=4 python cub-200/cub_binary_classify.py  # binary classification

CUDA_VISIBLE_DEVICES=4 python cub-200/cub_extract_feature_for_reranking.py  # sampling for reranking
CUDA_VISIBLE_DEVICES=4,5 python cub-200/cub_reranking.py  # reranking