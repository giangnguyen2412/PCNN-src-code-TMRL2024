import numpy as np

# Set the paths and filenames
folder_path = "/home/giang/Downloads/datasets/train_5k9_top5"
dict_path = "../faiss/cub/top5_NeurIPS_Finetuning_faiss_train_5k9_top1_HP_MODEL1_HP_FE.npy"

# Load the dictionary
file_dict = np.load(dict_path, allow_pickle=True).item()

import os
from shutil import copyfile

cnt = 0
# Function to recursively find and print JPG files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".jpg"):
            file_path = os.path.join(root, file)

            for i in range(5):
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
