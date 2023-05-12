import torch
import torch.nn as nn

import numpy as np
from datasets import ImageFolderForNNs
from helpers import HelperFunctions
import os
import glob

filename = 'faiss/cub/NeurIPS_Finetuning_faiss_CUB_train_aug_top1_HP_MODEL1_HP_FE.npy'
kbc = np.load(filename, allow_pickle=True, ).item()

HelperFunctions = HelperFunctions()
