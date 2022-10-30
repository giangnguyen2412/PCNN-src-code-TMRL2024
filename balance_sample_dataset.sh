rm -rf /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Wrong/Easy -s 8734 -o /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Wrong/Medium -s 16513 -o /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Wrong/Hard -s 3647 -o /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Correct/Easy -s 20000 -o /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Correct/Medium -s 5000 -o /home/giang/Downloads/datasets/SDogs_train
sh random_sample_dataset_v2.sh -d /home/giang/Downloads/RN18_dataset_Dog_train/Correct/Hard -s 3703 -o /home/giang/Downloads/datasets/SDogs_train