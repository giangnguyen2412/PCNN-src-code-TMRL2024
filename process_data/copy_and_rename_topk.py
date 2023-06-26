import numpy as np

# Set the paths and filenames
folder_path = "/home/giang/Downloads/datasets/CUB_train_all_top10"
dict_path = "../faiss/cub/top10_enriched_NeurIPS_Finetuning_faiss_CUB_train_all_top1_HP_MODEL1_HP_FE.npy"

folder_path = "/home/giang/Downloads/datasets/Dogs_train_top4"
dict_path = "../faiss/dogs/faiss_SDogs_train_RN34_top1.npy"


# Load the dictionary
file_dict = np.load(dict_path, allow_pickle=True).item()

import os
from shutil import copyfile

cnt = 0
# Function to recursively find and print JPG files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # if file.lower().endswith(".jpg"):
        if file.lower().endswith(".jpeg"):
            file_path = os.path.join(root, file)

            for i in range(4):
                if i == 0:
                    for j in range(4):  # Make up 5 NN sets from top-1 predictions
                        crt_file = 'Correct_{}_{}_'.format(i, j) + file
                        if crt_file in file_dict:
                            src_path = file_path
                            dst_path = os.path.join(root, crt_file)
                            copyfile(src_path, dst_path)
                            cnt += 1

                        wrong_file = 'Wrong_{}_{}_'.format(i, j) + file
                        if wrong_file in file_dict:
                            src_path = file_path
                            dst_path = os.path.join(root, wrong_file)
                            copyfile(src_path, dst_path)
                            cnt += 1
                else:
                    crt_file = 'Correct_{}_'.format(i) + file
                    if crt_file in file_dict:
                        src_path = file_path
                        dst_path = os.path.join(root, crt_file)
                        copyfile(src_path, dst_path)
                        cnt +=1

                    wrong_file = 'Wrong_{}_'.format(i) + file
                    if wrong_file in file_dict:
                        src_path = file_path
                        dst_path = os.path.join(root, wrong_file)
                        copyfile(src_path, dst_path)
                        cnt +=1

            os.remove(file_path)

print(cnt)
print("Files renamed/copied successfully!")
