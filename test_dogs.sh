#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python stanford-dogs/dogs_extract_feature.py  # sampling for test set
python stanford-dogs/clean_dups.py
python stanford-dogs/augment_images.py
CUDA_VISIBLE_DEVICES=4 python stanford-dogs/dogs_binary_classify.py  # binary classification

CUDA_VISIBLE_DEVICES=4 python stanford-dogs/dogs_extract_feature_for_reranking.py  # sampling for reranking
CUDA_VISIBLE_DEVICES=4,5 python stanford-dogs/dogs_reranking.py  # reranking