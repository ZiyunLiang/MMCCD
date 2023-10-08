import random
import torch as th
import os

import numpy as np

from datasets.brats2021 import BraTS2021Dataset_Cyclic, get_brats2021_train_transform_abnormalty_train, get_brats2021_train_transform_abnormalty_test

def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = th.Generator()
g.manual_seed(0)


def get_data_loader(dataset, data_path, config, input, trans=None, split_set='train', generator=True):
    if dataset == 'brats':
        loader = get_data_loader_brats_cyclic(input, trans, data_path, config.score_model.training.batch_size, config.score_model.image_size,
                                           split_set=split_set)
    else:
        raise Exception("Dataset does exit")

    return get_generator_from_loader(loader) if generator else loader


def get_data_loader_brats_cyclic(input, trans, path, batch_size, image_size, split_set: str = 'train'):

    assert split_set in ["train", "test"]
    default_kwargs = {"drop_last": False, "batch_size": batch_size, "pin_memory": False, "num_workers": 8,
                      "prefetch_factor": 8, "worker_init_fn": seed_worker, "generator": g, }
    if split_set == "test":
        patient_dir = os.path.join(path, 'test')

        default_kwargs["shuffle"] = False
        default_kwargs["num_workers"] = 8
        infer_transforms = get_brats2021_train_transform_abnormalty_test(image_size)
        dataset = BraTS2021Dataset_Cyclic(
            data_root=patient_dir,
            mode='test',
            input_mod=input,
            transforms=infer_transforms)
    else:
        patient_dir = os.path.join(path, 'train')
        default_kwargs["shuffle"] = True
        default_kwargs["num_workers"] = 4
        train_transforms = get_brats2021_train_transform_abnormalty_train(image_size)
        dataset = BraTS2021Dataset_Cyclic(
            data_root=patient_dir,
            mode='train',
            input_mod=input,
            trans_mod=trans,
            transforms=train_transforms)

    print(f"dataset lenght: {len(dataset)}")
    return th.utils.data.DataLoader(dataset, **default_kwargs)


def get_generator_from_loader(loader):
    while True:
        yield from loader
