import numpy as np
import os

cnt = 0
filename = 'faiss/advising_process_CUB_val_top1_HP_MODEL1_HP_FE.npy'
cnt_dict = {}
kbc = np.load(filename, allow_pickle=True, ).item()
for k, v in kbc.items():
    if v['Label'] == 1:
        cnt+=1

    k_base_name = k.split('_')
    k_base_name = ('_').join(k_base_name[3:])

    for nn in v['NNs']:
        base_name = os.path.basename(nn)
        if base_name in k:
            print("sth wrong")
            print(v)
            print(k)
            # exit(-1)
print(cnt)
print(len(kbc))
print(cnt*100/len(kbc))


# pass
# import os
#
# folder1 = "/home/giang/Downloads/nabirds_exact-match_split_small_50/train"
# folder2 = "/home/giang/Downloads/nabirds_exact-match_split_small_50/test"
#
# jpg_files1 = set()
# jpg_files2 = set()
#
# # Collect JPG files in folder1
# for root, _, files in os.walk(folder1):
#     for file in files:
#         # if file.lower().endswith(".JPEG"):
#         jpg_files1.add(file)
#
# # Collect JPG files in folder2 and check for overlaps
# overlapping_files = 0
# for root, _, files in os.walk(folder2):
#     for file in files:
#         # if file.lower().endswith(".JPEG"):
#         if file in jpg_files1:
#             overlapping_files += 1
#
# print("Number of overlapping JPG files:", overlapping_files)
