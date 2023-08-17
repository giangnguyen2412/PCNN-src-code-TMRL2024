import numpy as np
import os

cnt = 0
overlap = 0
crt_cnt = 0
filename = 'faiss/cub/top15_k1_enriched_NeurIPS_Finetuning_faiss_train5k7_top1_HP_MODEL1_HP_FE.npy'
cnt_dict = {}

kbc = np.load(filename, allow_pickle=True, ).item()
new_kbc = dict()
for k, v in kbc.items():
    if v['label'] == 1:
        cnt+=1

    if '_0_0_' in k:
        if v['label'] == 1:
            crt_cnt += 1

    k_base_name = k.split('_')
    k_base_name = ('_').join(k_base_name[3:])

    for nn in v['NNs']:
        base_name = os.path.basename(nn)
        if base_name in k:
            print("sth wrong")
            print(v)
            print(k)
            overlap += 1
            break
        else:
            new_kbc[k] = v
print(cnt)
print(cnt*100/len(kbc))
print(len(kbc))
print(overlap)
print(len(new_kbc))
# np.save(filename, new_kbc)
#

# Initialize a list to hold the accuracy at each threshold
# accuracies = []
#
# data_dict = np.load('SDogs_preds_dict.npy', allow_pickle=True).item()
# # Take only the first 1000 entries
# new_first_1000_entries = list(data_dict.items())[:1000]
#
# # Iterate through each threshold from 0.05 to 1.0 with a step of 0.05
# for threshold in np.arange(0.05, 1.05, 0.05):
#     # For each threshold, label the sample as True if the confidence is larger than or equal to the threshold, and False otherwise
#     predicted_labels = [entry[1]['confidence'] >= threshold for entry in new_first_1000_entries]
#
#     # The ground truth labels are stored in preds_dict[img_name]['correctness']
#     ground_truth_labels = [entry[1]['correctness'] for entry in new_first_1000_entries]
#
#     # The accuracy at each threshold is the number of correct labels (where the predicted label matches the ground truth) divided by the total number of samples
#     accuracy = sum(
#         predicted == ground_truth for predicted, ground_truth in zip(predicted_labels, ground_truth_labels)) / len(
#         new_first_1000_entries)
#     accuracies.append(accuracy)
#
# thresholds = np.arange(0.05, 1.05, 0.05)
# accuracies_dict = dict(zip(thresholds, accuracies))
# print(accuracies_dict)

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
