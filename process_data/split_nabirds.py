import os
import shutil

# Define the source and destination paths
source_folder = '/home/lab/datasets/nabirds/images'
split_file = '/home/lab/datasets/nabirds/train_test_split.txt'
destination_folder = '/home/giang/Downloads/nabirds_split_small/'

# Create the destination folders if they don't exist
os.makedirs(destination_folder + '0', exist_ok=True)
os.makedirs(destination_folder + '1', exist_ok=True)

# Read the split file into a dictionary
split_dict = {}
with open(split_file, 'r') as file:
    for line in file:
        basename, label = line.strip().split(' ')
        basename = basename.replace('-', '')
        if label not in split_dict:
            split_dict[label] = [basename]
        else:
            split_dict[label].append(basename)

random_classes = ["0295", "0341", "0374", "0459", "0492", "0525", "0558", "0628", "0661", "0753", "0786", "0819",
                  "0852",
                  "0885", "0918", "0951", "0984",
                  "0296", "0342", "0375", "0460", "0493", "0526", "0559", "0629", "0662", "0754", "0787", "0820",
                  "0853",
                  "0886", "0919", "0952", "0985",
                  "0297", "0343", "0376", "0461", "0494", "0527", "0560", "0630", "0663", "0755", "0788", "0821",
                  "0854",
                  "0887", "0920", "0953"]

overlapping_classes = ['0335', '0766', '0393', '1004', '0871', '0605', '0993', '1006', '0779', '0396', '0966', '0398',
                       '0954',
                       '0749', '0897', '0641', '0944', '0634', '0344', '0866', '0972', '0917', '0932', '0553', '0876',
                       '0867',
                       '0860', '0606', '0337', '0906', '0896', '0769', '0561', '0760', '0858', '0746', '0762', '0853',
                       '0788',
                       '0877', '0864', '0975', '0775', '0808', '0879', '0830', '0782', '0859', '0315', '0895']

overlapping_exact_classes = ['0879', '0911', '0828', '0884', '0853', '0895', '0533', '0875', '0897', '0865', '0802',
                             '0939',
                             '0900', '0864', '0868',
                             '0910', '0831', '0896', '0947', '0449', '0936', '0887', '0807', '0923', '0808', '0810',
                             '0870',
                             '0876', '0859', '0903',
                             '0858', '0479', '0915', '0450', '0803', '0862', '0827', '0866', '0830', '0474', '0476',
                             '0902',
                             '0941', '0534', '0886',
                             '0824', '0917', '0898', '0549', '0877']

classes = overlapping_exact_classes

# Iterate over JPEG files in the original folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.JPG'):
            basename = os.path.splitext(file)[0]
            label = os.path.basename(root)
            # Just subset 50 classes
            if label not in classes:
                continue
            # Check if the basename exists in the split dictionary
            if basename in split_dict['0']:
                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.join(destination_folder + '0', label), exist_ok=True)

                # Generate the source and destination paths for the image
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder + '0', label, file)

                # Copy the image to the appropriate destination folder
                shutil.copy2(source_path, destination_path)

            elif basename in split_dict['1']:
                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.join(destination_folder + '1', label), exist_ok=True)

                # Generate the source and destination paths for the image
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder + '1', label, file)

                # Copy the image to the appropriate destination folder
                shutil.copy2(source_path, destination_path)
            else:
                print(file)
        else:
            print(file)

print('Image splitting completed!')
