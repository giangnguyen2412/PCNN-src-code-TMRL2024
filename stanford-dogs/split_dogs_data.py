import os
import shutil
import scipy.io

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Paths to the dataset and .mat files
dataset_root = '/home/giang/Downloads/Stanford_Dogs_dataset'
images_dir = os.path.join(dataset_root, 'Images')
train_mat_path = os.path.join(dataset_root, 'train_list.mat')
test_mat_path = os.path.join(dataset_root, 'test_list.mat')

# Load .mat files
train_mat = scipy.io.loadmat(train_mat_path)
test_mat = scipy.io.loadmat(test_mat_path)

# Create train and test directories
train_dir = os.path.join(dataset_root, 'train')
test_dir = os.path.join(dataset_root, 'test')
create_dir_if_not_exists(train_dir)
create_dir_if_not_exists(test_dir)

# Function to copy images to train/test directories
def copy_images(file_list, source_dir, dest_dir):
    for item in file_list:
        file_rel_path = item[0][0]  # Relative path of the file
        source_path = os.path.join(source_dir, file_rel_path)
        dest_path = os.path.join(dest_dir, file_rel_path)
        dest_subdir = os.path.dirname(dest_path)

        # Create subdirectory if it doesn't exist
        create_dir_if_not_exists(dest_subdir)

        # Copy the file
        shutil.copy2(source_path, dest_path)

# Copy images to train and test directories
copy_images(train_mat['file_list'], images_dir, train_dir)
copy_images(test_mat['file_list'], images_dir, test_dir)

print("Images have been successfully copied to train and test directories.")
