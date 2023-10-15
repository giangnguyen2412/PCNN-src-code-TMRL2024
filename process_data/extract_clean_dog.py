from helpers import HelperFunctions
import os
import json
from shutil import copyfile

HelperFunctions = HelperFunctions()

real_json = open("reassessed-imagenet/real.json")
real_ids = json.load(real_json)
real_labels = {
            f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
            for i, labels in enumerate(real_ids)
            }
cnt_dict = {}
clean_dog_cnt = 0

def find_parent_folder(filename, parent_dir):
    for root, dirs, files in os.walk(parent_dir):
        if filename in files:
            return os.path.basename(root)
    return None

dogs_id = HelperFunctions.load_imagenet_dog_label()
for key, val in real_labels.items():
    num_class = len(val)
    if num_class in cnt_dict:
        cnt_dict[num_class] += 1
    else:
        cnt_dict[num_class] = 1
    # Choose clear images
    if num_class == 1:
        classs_id = val[0]
        class_wnid = HelperFunctions.key_list[classs_id]

        src_dir = f'{RunningParams.parent_dir}/datasets/imagenet1k-val'
        src_wnid = find_parent_folder(key, src_dir)
        src_path = os.path.join(src_dir, src_wnid, key)

        dst_dir = f'{RunningParams.parent_dir}/datasets/clean_dog'
        HelperFunctions.check_and_mkdir(os.path.join(dst_dir, class_wnid))
        dst_path = os.path.join(dst_dir, class_wnid, key)
        if class_wnid in dogs_id:
            clean_dog_cnt += 1
            copyfile(src_path, dst_path)
print(clean_dog_cnt)
print(cnt_dict)