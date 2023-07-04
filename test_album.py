from scipy.io import loadmat
import os
import shutil


def check_and_mkdir(f):
    if not os.path.exists(f):
        os.mkdir(f)
    else:
        pass


def check_and_rm(f):
    if os.path.exists(f):
        shutil.rmtree(f)
    else:
        pass

root = '/home/giang/Downloads/advising_network/stanford-dogs/data/downloads/Images'
train_dir = '/home/giang/Downloads/advising_network/stanford-dogs/data/downloads/train/'
test_dir = '/home/giang/Downloads/advising_network/stanford-dogs/data/downloads/test/'

check_and_mkdir(train_dir)
check_and_mkdir(test_dir)

train_list = loadmat("/home/giang/Downloads/advising_network/stanford-dogs/data/train_list.mat")
for f in train_list['file_list']:
    class_name = f[0][0].split('/')[0]
    file_name = f[0][0].split('/')[1]
    src = os.path.join(root, f[0][0])

    check_and_mkdir(os.path.join(train_dir, class_name))
    dst = os.path.join(train_dir, class_name, file_name)
    shutil.copyfile(src, dst)


test_list = loadmat("/home/giang/Downloads/advising_network/stanford-dogs/data/test_list.mat")
for f in test_list['file_list']:
    class_name = f[0][0].split('/')[0]
    file_name = f[0][0].split('/')[1]
    src = os.path.join(root, f[0][0])

    check_and_mkdir(os.path.join(test_dir, class_name))
    dst = os.path.join(test_dir, class_name, file_name)
    shutil.copyfile(src, dst)
