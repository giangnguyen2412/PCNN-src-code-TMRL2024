rm -rf /home/giang/Downloads/cub_train_test/RN50_dataset_CUB_Pretraining/train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN50_dataset_CUB_LOW/train/Wrong -s 1000 -o /home/giang/Downloads/cub_train_test/RN50_dataset_CUB_Pretraining/train
#sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset_Dog_val/Wrong/Medium -s 1000 -o /home/giang/Downloads/datasets/SDogs_val
#sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset_Dog_val/Wrong/Hard -s 1000 -o /home/giang/Downloads/datasets/SDogs_val
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN50_dataset_CUB_LOW/train/Correct -s 1000 -o /home/giang/Downloads/cub_train_test/RN50_dataset_CUB_Pretraining/train
#sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset_Dog_val/Correct/Medium -s 1000 -o /home/giang/Downloads/datasets/SDogs_val
#sh random_sample_dataset.sh -d /home/giang/Downloads/RN18_dataset_Dog_val/Correct/Hard -s 1000 -o /home/giang/Downloads/datasets/SDogs_val
#sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN34_SDogs_train/Wrong -s 19460 -o /home/giang/Downloads/datasets/Dogs_train
#sh random_sample_dataset.sh -d /home/giang/Downloads/RN34_SDogs_train/Correct -s 160 -o /home/giang/Downloads/datasets/Dogs_train


#sh random_sample_dataset.sh -d /home/giang/Downloads/datasets/CUB/test0 -s 5 -o /home/giang/Downloads/RN50_dataset_CUB_Pretraining/val
