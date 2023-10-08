import numpy as np
import random
import torch
import os
from pathlib import Path

patient_dir = os.path.join('/home/some5338/Documents/data/brats', 'BraTS2021_MiddleSlice_Train')

case_names = sorted(list(Path(patient_dir).iterdir()))
len_traning_set = round(0.7 * len(case_names))
len_testing_set = len(case_names) - round(0.7 * len(case_names))
training_names = case_names[:len_traning_set]
testing_names = case_names[len_testing_set:]
# with open("/home/some5338/Documents/codes/Diffusion_model/codes/datasets/brats_split_training.txt", 'w') as f:
with open("/home/some5338/Documents/data/brats/brats_split_training.txt", 'w') as f:
    for s in training_names:
        f.write(str(s) + '\n')
with open("/home/some5338/Documents/data/brats/brats_split_testing.txt", 'w') as f:
    for s in testing_names:
        f.write(str(s) + '\n')
print()