import numpy as np
import os
import random

filename = '/home/giang/Downloads/advising_network/faiss/cars/top10_k1_rn50_test-set_top1.npy'
kbc = np.load(filename, allow_pickle=True).item()

# Counters for labels
label_counts = {0: 0, 1: 0}

# Collect keys for entries with label 0
keys_to_remove = []

print("Initial total counts:", len(kbc))

for k, v in kbc.items():
    label_counts[v['label']] += 1

    if v['label'] == 0:
        keys_to_remove.append(k)

print("label_counts[0]:", label_counts[0])
print("label_counts[1]:", label_counts[1])

# Calculate how many 'label' 0 entries to remove to balance the dataset
num_to_remove = label_counts[0] - label_counts[1]

print("num_to_remove:", num_to_remove)

# Randomly select keys to remove to avoid bias
random.seed(42)  # For reproducibility
keys_to_remove = random.sample(keys_to_remove, num_to_remove)

# Remove selected keys from the dictionary
for key in keys_to_remove:
    del kbc[key]

# Save the balanced dictionary
np.save(filename, kbc)

# Print final counts for verification
final_counts = {0: 0, 1: 0}
for v in kbc.values():
    final_counts[v['label']] += 1

print("Final counts:", final_counts)
