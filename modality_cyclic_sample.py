"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import sys
import argparse
import random

import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path
from skimage import morphology
from sklearn.metrics import roc_auc_score, jaccard_score

from datasets import loader
from configs import get_config
from utils import logger
from utils.script_util import create_gaussian_diffusion, create_score_model
from utils.binary_metrics import assd_metric, sensitivity_metric, precision_metric
sys.path.append(str(Path.cwd()))



def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0.5)
    pred = pred.astype(int)
    targs = targs.astype(int)
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    config = get_config.file_from_dataset(args.dataset)
    if args.experiment_name_forward != 'None':
        experiment_name = args.experiment_name_forward
    else:
        raise Exception("Experiment name does exit")
    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])
    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating loader...")

    test_loader = loader.get_data_loader(args.dataset, args.data_dir, config, args.input, split_set='test',
                                          generator=False)
    logger.log("creating model and diffusion...")
    if args.model_name == 'unet':
        image_level_cond_forward = False
        image_level_cond_backward = False
    elif args.model_name == 'diffusion':
        image_level_cond_forward = True
        image_level_cond_backward = False
    else:
        raise Exception("Model name does exit")
    diffusion = create_gaussian_diffusion(config, args.timestep_respacing)
    model_forward = create_score_model(config, image_level_cond_forward)
    model_backward = create_score_model(config, image_level_cond_backward)

    filename = "model100000.pt"
    with bf.BlobFile(bf.join(logger.get_dir(), filename), "rb") as f:
        model_forward.load_state_dict(
            th.load(f.name, map_location=th.device('cuda'))
        )
    model_forward.to(th.device('cuda'))
    experiment_name_backward = f.name.split(experiment_name)[0] + args.experiment_name_backward + f.name.split(experiment_name)[1]
    model_backward.load_state_dict(
        th.load(experiment_name_backward, map_location=th.device('cuda'))
    )
    model_backward.to(th.device('cuda'))

    if config.score_model.use_fp16:
        model_forward.convert_to_fp16()
        model_backward.convert_to_fp16()

    model_forward.eval()
    model_backward.eval()

    logger.log("sampling...")

    dice = np.zeros(100)
    auc = np.zeros(1)
    assd = np.zeros(1)
    sensitivity = np.zeros(1)
    precision = np.zeros(1)
    jaccard = np.zeros(1)

    num_batch = 0
    num_sample = 0
    img_true_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size))
    img_pred_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size))
    brain_mask_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels, config.score_model.image_size, config.score_model.image_size))
    test_data_seg_all = np.zeros((len(test_loader.dataset), config.score_model.num_input_channels,
                               config.score_model.image_size, config.score_model.image_size))
    for test_data_dict in enumerate(test_loader):
        model_kwargs = {}
        ### brats dataset ###
        if args.dataset == 'brats':
            test_data_input = test_data_dict[1].pop('input').cuda()
            test_data_seg = test_data_dict[1].pop('seg')
            brain_mask = test_data_dict[1].pop('brainmask')
            brain_mask = (th.ones(brain_mask.shape) * (brain_mask > 0)).cuda()
            test_data_seg = (th.ones(test_data_seg.shape) * (test_data_seg > 0)).cuda()

        sample_fn = (
                        diffusion.p_sample_loop
        )
        sample = sample_fn(
            model_forward, model_backward, test_data_input, num_batch,
            (test_data_seg.shape[0], config.score_model.num_input_channels, config.score_model.image_size,
             config.score_model.image_size),
            model_name=args.model_name,
            clip_denoised=config.sampling.clip_denoised,  # is True, clip the denoised signal into [-1, 1].
            model_kwargs=model_kwargs,  # reconstruction = True
            eta=config.sampling.eta,
            model_forward_name=args.experiment_name_forward,
            model_backward_name=args.experiment_name_backward,
            ddim=args.use_ddim

        )
        num_batch += 1
        img_true_all[num_sample:num_sample+test_data_input.shape[0]] = test_data_input.detach().cpu().numpy()
        img_pred_all[num_sample:num_sample+test_data_input.shape[0]] = sample.cpu().numpy()
        brain_mask_all[num_sample:num_sample+test_data_input.shape[0]] = brain_mask.cpu().numpy()
        test_data_seg_all[num_sample:num_sample+test_data_input.shape[0]] = test_data_seg.cpu().numpy()
        num_sample += test_data_input.shape[0]
    logger.log("all the confidence maps from the testing set saved...")
    if args.model_name == 'unet':
        error_map = normalize((img_true_all - img_pred_all) ** 2)
    elif args.model_name == 'diffusion':
        filename_mask = "mask_forward_"+args.experiment_name_forward+'_backward_'+args.experiment_name_backward+".pt"
        filename_x0 = "cyclic_predict_"+args.experiment_name_forward+'_backward_'+args.experiment_name_backward+".pt"
        with bf.BlobFile(bf.join(logger.get_dir(), filename_mask), "rb") as f:
            tensor_load_mask = th.load(f)
        with bf.BlobFile(bf.join(logger.get_dir(), filename_x0), "rb") as f:
            tensor_load_xpred = th.load(f)
        load_gt_repeat = np.expand_dims(img_true_all, axis=0).repeat(tensor_load_mask.shape[0], axis=0)
        error_map = (np.abs(tensor_load_xpred.numpy() - load_gt_repeat)) ** 2
        mean_error_map = np.sum(error_map * (1-tensor_load_mask.numpy()), 0) / np.sum((1-tensor_load_mask.numpy()), 0)
        error_map = normalize(np.where(np.isnan(mean_error_map), 0, mean_error_map))
    logger.log("finding the best threshold...")
    for thres in range(100):
        mask_inpaint_input = np.where(thres / 1000 < error_map, 1.0, 0.0) * brain_mask_all
        for num in range(len(test_loader.dataset)):
            selem = morphology.disk(4)
            shrunken_brain_mask = morphology.erosion(
                brain_mask_all[num, 0, :, :], selem)
            dice[thres] += dice_score(test_data_seg_all[num, 0, :, :],
                               (mask_inpaint_input[num, 0, :, :]*shrunken_brain_mask))

    dice = dice / len(test_loader.dataset)
    max_dice_index = np.argmax(dice)
    max_dice = dice[max_dice_index]
    logger.log("computing the matrixs...")
    mask_inpaint_input = (np.where(max_dice_index / 1000 < error_map, 1.0, 0.0) * brain_mask_all)
    for num in range(len(test_loader.dataset)):
        selem = morphology.disk(4)
        shrunken_brain_mask = morphology.erosion(brain_mask_all[num, 0, :, :], selem)
        pred_thre = mask_inpaint_input[num, 0, :,:] * shrunken_brain_mask
        assd += assd_metric(pred_thre, test_data_seg_all[num, 0, :, :])
        sensitivity += sensitivity_metric(pred_thre, test_data_seg_all[num, 0, :, :])
        precision += precision_metric(pred_thre, test_data_seg_all[num, 0, :, :])
        pixel_wise_gt = (test_data_seg_all[num, 0, :, :].reshape(1, -1))[0, :]
        jaccard += jaccard_score(pixel_wise_gt, (normalize(
            np.array(mask_inpaint_input[num, 0, :, :])) * shrunken_brain_mask).astype(
            int).reshape(1, -1)[0, :])
        selem = morphology.disk(4)
        shrunken_brain_mask = morphology.erosion(brain_mask_all[num, 0, :, :], selem)
        pixel_wise_cls = (error_map[num, 0, :, :] * shrunken_brain_mask).reshape(1, -1)[0, :]
        auc += roc_auc_score(pixel_wise_gt, pixel_wise_cls)

    auc = auc / len(test_loader.dataset)
    jaccard = jaccard / len(test_loader.dataset)
    assd = assd / len(test_loader.dataset)
    sensitivity = sensitivity / len(test_loader.dataset)
    precision = precision / len(test_loader.dataset)

    logger.log(f"dice: {max_dice}, auc: {auc}, jaccard: {jaccard}, assd: {assd}, sensitivity: {sensitivity}, precision: {precision}")


def reseed_random(seed):
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='brats')
    parser.add_argument("--input", help="input modality, choose from flair, t2, t1", type=str, default='flair')
    parser.add_argument("--data_dir", help="data directory", type=str, default='./datasets/data')
    parser.add_argument("--experiment_name_forward", help="forward model saving file name", type=str, default='unet_brats_flair_t2')
    parser.add_argument("--experiment_name_backward", help="backward model saving file name", type=str, default='unet_brats_t2_flair')
    parser.add_argument("--model_name", help="translated model: unet or diffusion", type=str, default='unet')
    parser.add_argument("--use_ddim", help="if you want to use ddim during sampling, True or False", type=str, default='True')
    parser.add_argument("--timestep_respacing", help="If you want to rescale timestep during sampling. enter the timestep you want to rescale the diffusion prcess to. If you do not wish to resale thetimestep, leave it blank or put 1000.", type=int,
                        default=100)

    args = parser.parse_args()
    print(args.dataset)
    main(args)
