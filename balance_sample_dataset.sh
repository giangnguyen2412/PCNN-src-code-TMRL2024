#!/bin/sh

rm -rf /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Correct/Easy -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Correct/Medium -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Correct/Hard -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Wrong/Easy -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Wrong/Medium -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_val/Wrong/Hard -s 1000 -o /home/giang/Downloads/datasets/balanced_val_dataset_6k