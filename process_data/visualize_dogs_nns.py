from datasets import ImageFolderForNNs
from helpers import HelperFunctions
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HelperFunctions = HelperFunctions()
#
filename = '../faiss/faiss_SDogs_val_check_RN34_top1.npy'
file_a = np.load(filename, allow_pickle=True, ).item()
cnt = 0
for key, value in file_a.items():
    cnt += 1
    print(key, value[0])
    img_list = [key] + value[0:3]
    titles = []
    for img in img_list:
        dir = os.path.dirname(img)
        titles.append(HelperFunctions.id_map[os.path.basename(dir)].split(',')[0])
    ################################################################
    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # Loop through the image paths and plot each image
    for i, (img_path, title) in enumerate(zip(img_list, titles)):
        img = mpimg.imread(img_path)
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')

    # Show the figure
    plt.show()
    pass
    ################################################################

