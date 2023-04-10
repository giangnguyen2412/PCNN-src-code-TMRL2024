#!/bin/bash

shuf -n 1100 -e /home/giang/Downloads/RN50_dataset_CUB_HP/merged/*/*.jpg | xargs -I{} mv {} /home/giang/Downloads/RN50_dataset_CUB_HP/train_tmp/
shuf -n 200 -e /home/giang/Downloads/RN50_dataset_CUB_HP/merged/*/*.jpg | xargs -I{} mv {} /home/giang/Downloads/RN50_dataset_CUB_HP/val_tmp/
