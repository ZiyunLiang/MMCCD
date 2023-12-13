# Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI

This repository is the official pytorch implementation for paper: Liang et al., <a href="https://arxiv.org/abs/2308.16150">"Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI"</a>, Multiscale Multimodal Medical Imaging - MICCAI, 2023.


## Introduction:

[//]: # (Reconstruction based anomaly segmentation methods work by training the model to reconstruct in-distribution data and detects anomaly by the large reconstruction error. )

[//]: # (However, they are generic functions and may generalize to anomalies. )
In this paper, we propose a novel anomaly detection method by learning tissue-specific translation mapping functions. 
The intensities of tissues in different modalities is a unique characteristic of the tissue. Therefore, the translation functions of 'known' tissues among different modalities is not transferable to 'unknown' tissues, making modality translation an ideal choice for anomaly detection.
Our translation function is trained on in-distribution training data, and the anomalies are detected by the high translation error during inference.
Cyclic translation is further proposed so that only single modality data is needed during inference, while multiple modalities are needed for training. In this repository, UNet is used as a basic translation model to prove this idea. 
![Image text](https://github.com/ZiyunLiang/MMCCD/blob/master/img/img1.png)

Furthermore, Masked Conditional Diffusion Model is implemented as the forward translation model to show that diffusion model based inpainting can further improve the anomaly segmentation performance.
![Image text](https://github.com/ZiyunLiang/MMCCD/blob/master/img/img2.png)

## Usage:

### 1. preparation
**1.1 Environment**
**1.1 Environment**
We recommand you using conda for installing the depandencies.
The following command will help you create a new conda environment will all the required libraries installed: 
```
conda env create -f environment.yml
conda activate MMCCD
```
For manualy installing packages:
- `Python`                 3.9
- `torch`                   1.13.0
- `blobfile`                2.0.2
- `numpy`                   1.23.0
- `scikit-learn`            1.3.1
- `scikit-image`            0.22.0
- `scipy`                   1.10.0
- `tqdm`                    4.66.1
- `nibabel`                 5.1.0
- `monai`                   1.2.0
- `tensorboard`            2.14.1
- -`ml_collections`         0.1.1

The project can be cloned using:
```
git clone https://github.com/ZiyunLiang/MMCCD.git
```
**1.2 Dataset Download**\
Download the BraTS2021 training data from <a href="http://www.braintumorsegmentation.org/">BraTS2021 official website</a> and unzip it to the folder `./datasets/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021`.
The `brats_split_training.txt` and `brats_split_testing.txt` in `./dataset` are the split of the training data and testing dataset used in the paper. 

**1.3 Data Preprocessing**\
We preprocess the 3D data and save the normalized 2D slices for training to save the data loading time.
We have the data preprocessing script in `./datasets/brats_preprocess.py`. 
From every 3D image, we extract slices 70 to 90, that mostly capture the central part of the brain. For model training, from the slices extracted from the training subjects, we only use those that
do not contain any tumors. And for testing subjects, the slide with the biggest tumor is selected. Data is normalized during the preprocessing process. The (percentage_upper, percentage_lower) intensity of the image is normalized, and the parameters can be modified by the command line argument.
The command line arguments when running the script:
  - `--modality` Allows you to choose which modality you want to preprocess and save. Multiple modalities can be added, but should be separated with ',' without space. Default: 't2,flair,t1'
  - `--data_dir` The directory of the already downloaded brats data. Default: './datasets/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
  - `--output_dir` The directory to save the preprocessed data. Default: './datasets/data'
  - `--percentage_upper` When normalizing the data, the upper percentage of data intensity for normalization, the value should be between[0-1]. Default: 0.9
  - `--percentage_lower` When normalizing the data, the lower percentage of data intensity for normalization, the value should be between[0-1]. Default: 0.02

Below is an example script for preprocessing data (all arguments are set to default).
```
python ./datasets/brats_preprocess.py 
```

### 2. Training

The  training script is in `modality_cyclic_train.py`. The arguments for this script are:
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--dataset` Allows you to choose the dataset that you want to use for this experiment, only BraTS is included in this implementation. Feel free to test it on your own dataset. Default: 'brats'
  - `--data_dir` The directory of the already preprocessed brats data. Default: './datasets/data'
  - `--input` The input modality in the cyclic translation process, which is the only modality used for testing. Default: 'flair'
  - `--trans` The translated modality in the cyclic translation process, which is the intermediate modality in the cyclic process. Default: 't2'
  - `--experiment_name` The file name for saving the model. Default: 'None'
  - `--model_name` Which model is selected for the translation process. Only unet and diffusion model are included in this implementation, feel free to add your own translation model. 
The input should be either 'unet' or 'diffusion'. In our implementation, if you want to try out the model using basic unet ('Cyclic UNet'), then both forward and backward model should be set to 'unet'. 
If you want to try out translation with diffusion model (MMCCD), then the model for forward process should be set to 'diffusion', and backward process should be set to 'unet'. 
 Default: 'unet'

The other hyperparameters used for training are in `./config/brats_config.py`. Note that the model needs to be trained twice for the forward and backward process.
Below is an example script for training the forward and backward unet model with our default settings on cyclic UNet:
```
python modality_cyclic_train.py --input flair --trans t2 --model_name unet 
```
```
python modality_cyclic_train.py --input t2 --trans flair --model_name unet 
```
Here is another example script for training the model for MMCCD (using diffusion model) which input modality is flair, and translated modality is t1. The model is saved to the file 'diffusion_brats_flair_t1':
```
python modality_cyclic_train.py --input flair --trans t1 --model_name diffusion 
```
```
python modality_cyclic_train.py --input t1 --trans flair --model_name unet 
```

### 3. Testing 
The testing script is in `modality_cyclic_test.py`.
The arguments for this script are:
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--dataset` Allows you to choose the dataset that you want to use for this experiment, only BraTS is included in this implementation. Feel free to test it on your own dataset. Default: 'brats'
  - `--data_dir` The directory of the already preprocessed brats data. Default: './datasets/data'
  - `--experiment_name_forward` The file name for trained forward model so that we can load the saved model. Default: 'unet_forward_brats_t2_flair'
  - `--experiment_name_backward` The file name for trained backward model so that we can load the saved model. Default: 'unet_backward_brats_t2_flair'
  - `--model` which model is used for the translation process, only two models are included in this implementation, 
the input should be either 'unet' or 'diffusion'. If 'unet' is chosen, it will perform the testing for 'Cyclic UNet', where both forward and backward model is UNet. If 'diffusion' is chosen, it will perform the testing for 'MMCCD', where the forward model is diffusion and backward model is unet. Default: 'unet'
  - `--use_ddim` If you want to use ddim during sampling. True or False. Default: 'True'
  - `--timestep_respacing` 
The other hyperparameters used for testing are in `./config/brats_config.py`.

Below is an example script for testing with our default settings using Cyclic UNet:
```
python modality_cyclic_test.py 
```
Here is another example script for testing the model for MMCCD (using diffusion model). The input modality is flair, 
and the forward model is loaded from the file 'diffusion_brats_flair_t2', the backward model is loaded from the file 'unet_brats_t2_flair':
```
python modality_cyclic_test.py --experiment_name_forward diffusion_brats_flair_t2 --experiment_name_backward unet_brats_t2_flair --dataset flair --model_name diffusion --use_ddim True
```
## Citation
If you have any questions, please contact Ziyun Liang (ziyun.liang@eng.ox.ac.uk) and I am happy to discuss more with you. 
If you find this work helpful for your project, please give it a star and a citation. 
We greatly appreciate your acknowledgment.
```
@article{liang2023modality,
  title={Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI},
  author={Liang, Ziyun and Anthony, Harry and Wagner, Felix and Kamnitsas, Konstantinos},
  journal={arXiv preprint arXiv:2308.16150},
  year={2023}
}
```

## Acknowledgements
We would like to acknowledge the diffusion model implementation from 
https://github.com/openai/guided-diffusion/tree/main/guided_diffusion, where this repository is based upon.

## License
This project is licensed under the terms of the MIT license.
MIT License

Copyright (c) 2023 Ziyun Liang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
