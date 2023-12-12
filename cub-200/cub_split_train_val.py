import shutil
import os

def read_split_file(split_file_path):
    split_data = {}
    with open(split_file_path, 'r') as file:
        for line in file:
            image_id, data_id = map(int, line.strip().split())
            split_data[image_id] = data_id
    return split_data

def read_images_file(images_file_path):
    images_data = {}
    with open(images_file_path, 'r') as file:
        for line in file:
            image_id, image_path = map(str, line.strip().split())
            images_data[int(image_id)] = image_path
    return images_data

def split_dataset(original_dataset_path, output_train_path, output_test_path, split_file_path, images_file_path):
    # Read split and images data
    split_data = read_split_file(split_file_path)
    images_data = read_images_file(images_file_path)

    # Create output directories if they don't exist
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    with open(class_folder_file_path, 'r') as f:
        class_folders = [line.strip().split()[1] for line in f]

    # Create class folders in output_train_path
    for folder in class_folders:
        os.makedirs(os.path.join(output_train_path, folder), exist_ok=True)

    # Create class folders in output_test_path
    for folder in class_folders:
        os.makedirs(os.path.join(output_test_path, folder), exist_ok=True)

    # Copy images to respective directories based on split information
    for image_id, data_id in split_data.items():
        image_path = images_data.get(image_id)
        if image_path:
            output_directory = output_test_path if data_id == 0 else output_train_path
            shutil.copy(os.path.join(original_dataset_path, image_path), os.path.join(output_directory, image_path))

# Example usage
original_dataset_path = "/home/daoduyhung/data/CUB_200_2011/images/"
output_train_path = "/home/daoduyhung/data/CUB_200_2011/train/"
output_test_path = "/home/daoduyhung/data/CUB_200_2011/test/"
split_file_path = "/home/daoduyhung/data/CUB_200_2011/train_test_split.txt"
images_file_path = "/home/daoduyhung/data/CUB_200_2011/images.txt"
class_folder_file_path = "/home/daoduyhung/data/CUB_200_2011/classes.txt"

split_dataset(original_dataset_path, output_train_path, output_test_path, split_file_path, images_file_path)



def count_images_in_folder(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count

print(f"Number of images in output_train_folder: { count_images_in_folder(output_train_path)}")
print(f"Number of images in output_test_folder: { count_images_in_folder(output_test_path)}")
