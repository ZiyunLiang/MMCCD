'this file is for transfering 3D BRaTs MRI to 2D Slices of jpg image for training'
import os
import argparse

import numpy as np
import nibabel as nib

def nii2np_train(img_root, img_name, upper_per, lower_per, output_root_train=None, modality=None):
    # img_name = (img_path.split('/')[-1]).split('.')[0]
    modality = modality.split(',')
    '''generate image for each modality'''
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
        img = nib.load(img_file)
        img = (img.get_fdata())

        '''normalize the [lower_per, lower_per] of the brain to [-3,3]'''
        img_original = img
        perc_upper = ((img > 0).sum() * (1 - upper_per)) / (img.shape[0] * img.shape[1] * img.shape[2])  # find the proportion of top (upper_per)% intensity of the brain within the whole 3D image
        perc_lower = ((img > 0).sum() * lower_per) / (img.shape[0] * img.shape[1] * img.shape[2])
        upper_value = np.percentile(img, (1 - perc_upper) * 100)
        lower_value = np.percentile(img, perc_lower * 100)
        img_half = (upper_value - lower_value) / 2
        img = (img - img_half) / (upper_value - lower_value) * 6   # normalize the [lower_per, upper_per] of the brain to [-3,3]

        img_file_label = os.path.join(img_root, img_name, img_name + '_' + 'seg' + '.nii.gz')
        img_label = nib.load(img_file_label)
        img_label = (img_label.get_fdata())
        img_label = img_label.astype(np.uint8)

        '''only convert the healthy middle slice for training'''
        for i in range(20):
            slice = i + 70
            if np.sum(img_label[:,:,slice]) == 0:
                dirs_mod = os.path.join(output_root_train, modality[mod_num])
                if not os.path.exists(dirs_mod):
                    os.makedirs(dirs_mod)
                filename = os.path.join(dirs_mod, img_name + '_' + modality[mod_num] + '_' + str(slice))
                img_slice = img[:, :, slice]
                np.save(filename, img_slice)

                if mod_num == 0:
                    dirs_seg = os.path.join(output_root_train, 'seg')
                    if not os.path.exists(dirs_seg):
                        os.makedirs(dirs_seg)
                    filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + str(slice))
                    img_slice_seg = img_label[:, :, slice]
                    np.save(filename_seg, img_slice_seg)
                    dirs_brainmask = os.path.join(output_root_train, 'brainmask')
                    if not os.path.exists(dirs_brainmask):
                        os.makedirs(dirs_brainmask)
                    filename_brainmask = os.path.join(dirs_brainmask, img_name + '_brainmask_' + str(slice))
                    img_brainmask = (img_original > 0).astype(int)
                    img_slice_brainmask = img_brainmask[:, :, slice]
                    np.save(filename_brainmask, img_slice_brainmask)


def nii2np_test(img_root, img_name, upper_per, lower_per, output_root_test=None, modality=None):
    # img_name = (img_path.split('/')[-1]).split('.')[0]
    modality = modality.split(',')
    '''generate image for each modality'''
    for mod_num in range(len(modality)):
        img_file = os.path.join(img_root, img_name, img_name + '_' + modality[mod_num] + '.nii.gz')
        img = nib.load(img_file)
        img = (img.get_fdata())
        img_original = img

        '''normalize the [lower_per, lower_per] of the brain to [-3,3]'''
        perc_upper = ((img > 0).sum() * (1 - upper_per)) / (img.shape[0] * img.shape[1] * img.shape[
            2])  # find the proportion of top (upper_per)% intensity of the brain within the whole 3D image
        perc_lower = ((img > 0).sum() * lower_per) / (img.shape[0] * img.shape[1] * img.shape[2])
        upper_value = np.percentile(img, (1 - perc_upper) * 100)
        lower_value = np.percentile(img, perc_lower * 100)
        img_half = (upper_value - lower_value) / 2
        img = (img - img_half) / (upper_value - lower_value) * 6  # normalize the bottom (1-x%) of the brain to [-3,3]

        img_file_label = os.path.join(img_root, img_name, img_name + '_' + 'seg' + '.nii.gz')
        img_label = nib.load(img_file_label)
        img_label = img_label.get_fdata()
        img_label = img_label.astype(np.uint8)


        '''find the slice with maximum tumor area for testing '''
        img_label = np.ones(img_label.shape) * (img_label > 0)
        if np.max(np.sum(img_label[:,:,70:90], axis=(0,1))) == 0:
            print('pass')
            pass
        else:
            slice = np.argmax(np.sum(img_label[:,:,70:90], axis=(0,1))) + 70

            dirs_mod = os.path.join(output_root_test, modality[mod_num])
            if not os.path.exists(dirs_mod):
                os.makedirs(dirs_mod)
            filename = os.path.join(dirs_mod, img_name + '_' + modality[mod_num] + '_' + str(slice))
            img_slice = img[:, :, slice]
            np.save(filename, img_slice)

            if mod_num == 0:
                dirs_seg = os.path.join(output_root_test, 'seg')
                if not os.path.exists(dirs_seg):
                    os.makedirs(dirs_seg)
                filename_seg = os.path.join(dirs_seg, img_name + '_seg_' + str(slice))
                img_slice_seg = img_label[:, :, slice]
                np.save(filename_seg, img_slice_seg)
                dirs_brainmask = os.path.join(output_root_test, 'brainmask')
                if not os.path.exists(dirs_brainmask):
                    os.makedirs(dirs_brainmask)
                filename_brainmask = os.path.join(dirs_brainmask, img_name + '_brainmask_' + str(slice))
                img_brainmask = (img_original > 0).astype(int)
                img_slice_brainmask = img_brainmask[:, :, slice]
                np.save(filename_brainmask, img_slice_brainmask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the directory in which the data is stored", type=str, default='./data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData')
    parser.add_argument("--output_dir", help="the directory to store the preprocessed data", type=str, default='./data')
    parser.add_argument("--modality", help="The generated modality, like 't1', 't2', or 'flair'. Multi-modality separate by ',' without space, like 't1,t2'", type=str,
                        default='t2,flair,t1')
    parser.add_argument("--upper_per", help="The upper percentage of brain area to be normalized, the value needs to be within [0-1], like 0.9", type=float, default=0.9)
    parser.add_argument("--lower_per", help="The lower percentage of brain area to be normalized, the value needs to be within [0-1], like 0.02", type=float, default=0.02)
    args = parser.parse_args()
    img_root = args.data_dir
    img_output_root = args.output_dir
    img_output_root_train = os.path.join(img_output_root, 'train')
    img_output_root_test = os.path.join(img_output_root, 'test')
    train_txt = './brats_split_training.txt'
    test_txt = './brats_split_testing.txt'

    MOD = args.modality
    with open(train_txt) as file:
        for path in file:
            nii2np_train(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root_train=img_output_root_train, modality=MOD)
    with open(test_txt) as file:
        for path in file:
            nii2np_test(img_root, path[:-1], upper_per=args.upper_per, lower_per=args.lower_per, output_root_test=img_output_root_test, modality=MOD)