import numpy as np
import os
from params import RunningParams
from shutil import copyfile

RunningParams = RunningParams()

# Set the paths and filenames
folder_path = RunningParams.aug_data_dir
dict_path = RunningParams.faiss_npy_file

# Load the dictionary
file_dict = np.load(dict_path, allow_pickle=True).item()
cnt = 0

breakpoint()
# Function to recursively find and print JPG files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(RunningParams.extension):
            file_path = os.path.join(root, file)
            for i in range(RunningParams.Q):
                if i == 0:
                    for j in range(RunningParams.Q):  # Make up NN sets from top-1 predictions
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

            # Remove the original image
            os.remove(file_path)

print(f"{cnt} files copied successfully!")

