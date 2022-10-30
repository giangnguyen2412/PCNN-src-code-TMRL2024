import os
from shutil import copyfile
import glob
imagenet_folders = glob.glob('/home/giang/Downloads/datasets/imagenet1k-val/*')

# Get IDs of 120 dog breeds
def load_imagenet_dog_label():
    count = 0
    dog_id_list = list()
    input_f = open('/home/giang/Downloads/SDogs_dataset/dog_type.txt')
    for line in input_f:
        dog_id = (line.split('-')[0])
        dog_id_list.append(dog_id)
    return dog_id_list

def check_and_mkdir(f):
    if not os.path.exists(f):
        os.mkdir(f)
    else:
        pass

dogs_id = load_imagenet_dog_label()
# print(dogs_id)
cnt = 0
for idx, imagenet_folder in enumerate(imagenet_folders):
    imagenet_id = imagenet_folder.split('imagenet1k-val/')[1]
    # if imagenet_id not in dogs_id:
    #     continue
    cnt += 1
    # print(imagenet_folder)
    imagenet_wnid = os.path.basename(imagenet_folder)
    dir = os.path.join('/home/giang/Downloads/datasets/SDogs_train', imagenet_wnid)
    check_and_mkdir(dir)
    # cmd = 'cp -r {} {}'.format(imagenet_folder, '/home/giang/Downloads/Dogs_dataset/val')
    # os.system(cmd)
print(cnt)

