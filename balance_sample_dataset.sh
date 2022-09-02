#!/bin/sh

rm -rf /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset/Correct/Easy -s 30 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset/Correct/Medium -s 30 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset/Correct/Hard -s 30000 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset/Wrong/Easy -s 30 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset/Wrong/Medium -s 30 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset/Wrong/Hard -s 30000 -o /home/giang/Downloads/datasets/balanced_train_dataset_180k