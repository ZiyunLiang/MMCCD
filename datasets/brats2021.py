import os

import numpy as np
import monai.transforms as transforms
from os.path import join
from pathlib import Path

from torch.utils.data import Dataset


def get_brats2021_train_transform_abnormalty_train(image_size):
    base_transform = get_brats2021_base_transform_abnormalty_train(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'trans', 'brainmask', 'seg']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_brats2021_base_transform_abnormalty_train(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'trans', 'brainmask', 'seg']),
        transforms.Resized(
            keys=['input', 'trans', 'brainmask', 'seg'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

def get_brats2021_train_transform_abnormalty_test(image_size):
    base_transform = get_brats2021_base_transform_abnormalty_test(image_size)
    data_aug = [
        transforms.EnsureTyped(
            keys=['input', 'brainmask', 'seg']),
    ]
    return transforms.Compose(base_transform + data_aug)

def get_brats2021_base_transform_abnormalty_test(image_size):

    base_transform = [
        transforms.AddChanneld(
            keys=['input', 'brainmask', 'seg']),
        transforms.Resized(
            keys=['input', 'brainmask', 'seg'],
            spatial_size=(image_size, image_size)),
    ]

    return base_transform

class BraTS2021Dataset_Cyclic(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod, trans_mod=None, transforms=None):
        super(BraTS2021Dataset_Cyclic, self).__init__()

        assert mode in ['train', 'test'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_mod = input_mod

        self.transforms = transforms
        self.case_names_input = sorted(list(Path(os.path.join(self.data_root, input_mod)).iterdir()))
        self.case_names_brainmask = sorted(list(Path(os.path.join(self.data_root, 'brainmask')).iterdir()))
        self.case_names_seg = sorted(list(Path(os.path.join(self.data_root, 'seg')).iterdir()))
        if mode == 'train':
            self.trans_mod = trans_mod
            self.case_names_trans = sorted(list(Path(os.path.join(self.data_root, trans_mod)).iterdir()))

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index].name
        name_brainmask = self.case_names_brainmask[index].name
        name_seg = self.case_names_seg[index].name
        base_dir_input = join(self.data_root, self.input_mod, name_input)
        base_dir_brainmask = join(self.data_root, 'brainmask', name_brainmask)
        base_dir_seg = join(self.data_root, 'seg', name_seg)
        input = np.load(base_dir_input).astype(np.float32)

        brain_mask = np.load(base_dir_brainmask).astype(np.float32)
        seg = np.load(base_dir_seg).astype(np.float32)
        if self.mode == 'train':
            name_trans = self.case_names_trans[index].name
            base_dir_trans = join(self.data_root, self.trans_mod, name_trans)
            trans = np.load(base_dir_trans).astype(np.float32)
            item = self.transforms(
                {'input': input, 'trans': trans, 'brainmask': brain_mask, 'seg': seg})
        else:
            item = self.transforms(
                {'input': input, 'brainmask': brain_mask, 'seg': seg})

        return item

    def __len__(self):
        return len(self.case_names_input)