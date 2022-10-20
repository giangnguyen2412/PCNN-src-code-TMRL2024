rm -rf /home/giang/Downloads/datasets/random_train_dataset_400k
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Correct/Easy -s 83000 -o /home/giang/Downloads/datasets/random_train_dataset
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Correct/Medium -s 80000 -o /home/giang/Downloads/datasets/random_train_dataset
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Correct/Hard -s 37000 -o /home/giang/Downloads/datasets/random_train_dataset
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Wrong/Easy -s 89000 -o /home/giang/Downloads/datasets/random_train_dataset
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Wrong/Medium -s 79000 -o /home/giang/Downloads/datasets/random_train_dataset
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_train/Wrong/Hard -s 32000 -o /home/giang/Downloads/datasets/random_train_dataset