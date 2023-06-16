import pandas as pd

# Read the CSV files
overlapping_cub_narbirds_df = pd.read_csv('../overlapping_cub_narbirds.csv')
terminal_classes_df = pd.read_csv('../terminal_classes_without_reset_index.csv')

# Get the unique NABirds Classes from overlapping_cub_narbirds_df
unique_nabirds_classes = overlapping_cub_narbirds_df['NABirds Classes'].unique()

# Filter the terminal_classes_df based on matching label_name in NABirds Classes
filtered_ids = terminal_classes_df[terminal_classes_df['label_name'].isin(unique_nabirds_classes)]['id']

# Print the filtered IDs
print(filtered_ids)
print(len(filtered_ids))

overlapping_classes = filtered_ids.values.tolist()

sce_classes = [295, 463, 696, 314, 613, 316, 615, 318, 617, 319, 618, 320, 619, 321, 620, 322, 621,
                     323, 622, 324, 623, 326, 327, 625, 626, 329, 330, 331, 628, 629, 630, 333, 334, 632,
                     633, 338, 339, 637, 638, 357, 358, 359, 360, 361, 656, 657, 658, 659, 660, 365, 366,
                     664, 665]

scs_classes = [755, 763, 962, 970, 772, 979, 888, 891, 892, 893, 348, 349, 475, 648, 647, 783, 990, 799, 789, 790,
               918, 919, 996, 997, 926, 927, 931, 793, 1000, 345, 472, 644, 774, 981, 395, 402, 446, 447, 602, 609,
               953, 952, 949, 948, 940, 942, 943, 945, 1010, 975, 905, 768, 746]

overlappings = []
for overlapping_class in scs_classes:
    class_id = str(overlapping_class)
    if overlapping_class < 1000:
        class_id = '0' + class_id
    overlappings.append(class_id)

print(overlappings)
print(len(overlappings))

import random
random.seed(42)
random.shuffle(overlappings)
# print(overlappings[:50])
print(overlappings)