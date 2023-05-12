import os
import glob

# Define the path to the folder to remove empty subfolders from
folder_path = "/home/giang/Downloads/datasets/CUB_val_hard"

cnt = 0
# Function to remove empty subfolders recursively
def remove_empty_folders(folder_path):
    global cnt
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for folder in dirs:
            folder_to_remove = os.path.join(root, folder)
            jpg_files = glob.glob(os.path.join(folder_to_remove, '*.jpg'))
            # print(jpg_files)
            if not jpg_files:
                os.rmdir(folder_to_remove)
                cnt += 1
                # print("Removed empty folder:", folder_to_remove)
            else:
                pass

# Call the function to remove empty subfolders
remove_empty_folders(folder_path)
print("Removed {} empty folder:", cnt)