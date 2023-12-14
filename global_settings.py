import sys
import os

project_path = "/home/daoduyhung/hicehehe/xai/advising_network"

cub_train_path = "/home/daoduyhung/data/CUB_200_2011/train"
cub_test_path = "/home/daoduyhung/data/CUB_200_2011/test"
cub_full_path = "/home/daoduyhung/data/CUB_200_2011/images"

car_train_path = ""
car_test_path = ""

CUDA_VISIBLE_DEVICES = "0, 1"

CNN_ADVISING_NET = True # enable CNN Advising Net

EXTRACT_FEATURE_TEST_DATASET = False # set to 'train' or 'test'
