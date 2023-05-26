import numpy as np

cnt = 0
filename = 'faiss/cub/top5_NeurIPS_Finetuning_faiss_train_5k9_top1_HP_MODEL1_HP_FE.npy'
kbc = np.load(filename, allow_pickle=True, ).item()
for k, v in kbc.items():
    if v['label'] == 1:
        cnt+=1
print(cnt)
pass
#
# #####
#
# import os
# import shutil
#
# def split_folder(folder_path, folder_1_size):
#     # Create two new folders to hold the split data
#     folder_1 = os.path.join(dir_path, 'folder_1')
#     os.makedirs(folder_1, exist_ok=True)
#
#     # Get a list of subfolders
#     subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
#
#     # Iterate through each subfolder
#     for subfolder in subfolders:
#         subfolder_path = os.path.join(folder_path, subfolder)
#         jpg_files = [file for file in os.listdir(subfolder_path) if file.endswith('.jpg')]
#
#         # Move files to folder_1
#         for i in range(folder_1_size):
#             jpg_file = jpg_files[i]
#             source_file = os.path.join(subfolder_path, jpg_file)
#             destination_folder = folder_1
#             destination_file = os.path.join(destination_folder, subfolder, jpg_file)
#             os.makedirs(os.path.dirname(destination_file), exist_ok=True)
#             shutil.move(source_file, destination_file)
#
#     print("Folder split completed successfully.")
#
# # Specify the folder path and size for folder_1
# folder_path = '/home/giang/Downloads/datasets/test_4k7'
# dir_path = '/home/giang/Downloads/datasets'
# folder_1_size = 5
#
# # Call the function to split the folder
# split_folder(folder_path, folder_1_size)
