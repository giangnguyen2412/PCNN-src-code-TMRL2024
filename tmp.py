import glob
import os
test_path = '/home/giang/Downloads/RN50_dataset_CUB_LP/val'
test_samples1 = []
image_folders = glob.glob('{}/*'.format(test_path))
for i, image_folder in enumerate(image_folders):
    id = image_folder.split('val/')[1]
    image_paths = glob.glob(image_folder + '/*.*')
    for image in image_paths:
        test_samples1.append(os.path.basename(image))

test_path = '/home/giang/Downloads/RN50_dataset_CUB_HP_finetune_set/val'
test_samples2 = []
image_folders = glob.glob('{}/*'.format(test_path))
for i, image_folder in enumerate(image_folders):
    id = image_folder.split('val/')[1]
    image_paths = glob.glob(image_folder + '/*.*')
    for image in image_paths:
        test_samples2.append(os.path.basename(image))

print(len(set(test_samples1) & set(test_samples2)))

temp3 = []
for element in test_samples2:
    if element not in test_samples1:
        temp3.append(element)
print(len(temp3))
print(temp3)
