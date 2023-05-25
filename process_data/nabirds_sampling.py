import pandas as pd

# Read the CSV files
overlapping_cub_narbirds_df = pd.read_csv('overlapping_cub_narbirds.csv')
terminal_classes_df = pd.read_csv('terminal_classes_without_reset_index.csv')

# Get the unique NABirds Classes from overlapping_cub_narbirds_df
unique_nabirds_classes = overlapping_cub_narbirds_df['NABirds Classes'].unique()

# Filter the terminal_classes_df based on matching label_name in NABirds Classes
filtered_ids = terminal_classes_df[terminal_classes_df['label_name'].isin(unique_nabirds_classes)]['id']

# Print the filtered IDs
print(filtered_ids)
print(len(filtered_ids))

overlapping_classes = filtered_ids.values.tolist()

overlappings = []
for overlapping_class in overlapping_classes:
    class_id = str(overlapping_class)
    if overlapping_class < 1000:
        class_id = '0' + class_id
    overlappings.append(class_id)

print(overlappings)
print(len(overlappings))

import random
random.seed(42)
random.shuffle(overlappings)
print(overlappings[:50])