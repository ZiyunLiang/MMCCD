"""
Train a diffusion model on images.
"""
import time
import sys
import argparse
import os

from pathlib import Path
import torch as th

from configs import get_config
from utils import logger
from datasets import loader
from models.resample import create_named_schedule_sampler
from utils.script_util import create_gaussian_diffusion, create_score_model
from utils.train_util import TrainLoop
sys.path.append(str(Path.cwd()))


def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    time_load_start = time.time()
    config = get_config.file_from_dataset(args.dataset)

    if args.experiment_name != 'None':
        experiment_name = args.experiment_name
    else:
        experiment_name = args.training_process + '_' + args.dataset + '_' + args.input + '_' + args.trans

    logger.configure(Path(experiment_name)/"score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    if args.model_name == 'unet':
        image_level_cond = False
    elif args.model_name == 'diffusion':
        image_level_cond = True
    else:
        raise Exception("Model name does exit")

    logger.configure(Path(experiment_name) / "score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])
    diffusion = create_gaussian_diffusion(config, timestep_respacing=False)
    model = create_score_model(config, image_level_cond)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = th.device(config.device)
    model.to(device)

    logger.log(f"Model number of parameters {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(config.score_model.training.schedule_sampler, diffusion)

    if args.training_process == 'forward':
        input = args.input
        trans = args.trans
    elif args.training_process == 'backward':
        input = args.trans
        trans = args.input
    else:
        raise Exception("Training process does exit")

    logger.log("creating data loader...")
    train_loader = loader.get_data_loader(args.dataset, args.data_dir, config, input, trans, split_set='train', generator=True)
    time_load_end = time.time()
    time_load = time_load_end - time_load_start
    logger.log("data loaded: time ", str(time_load))
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=config.score_model.training.batch_size,
        lr=config.score_model.training.lr,
        ema_rate=config.score_model.training.ema_rate,
        log_interval=config.score_model.training.log_interval,
        save_interval=config.score_model.training.save_interval,
        use_fp16=config.score_model.training.use_fp16,
        fp16_scale_growth=config.score_model.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.score_model.training.weight_decay,
        lr_decay_steps=config.score_model.training.lr_decay_steps,
        lr_decay_factor=config.score_model.training.lr_decay_factor,
        iterations=config.score_model.training.iterations,
        num_samples=config.sampling.num_samples,
        num_input_channels=config.score_model.num_input_channels,
        image_size=config.score_model.image_size,
        clip_denoised=config.sampling.clip_denoised,
        use_ddim=False,
        device=device,
        args=args
    ).run_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='brats')
    parser.add_argument("--input", help="input modality, choose from flair, t2, t1", type=str, default='flair')
    parser.add_argument("--trans", help="translated modality, choose from flair, t2, t1", type=str, default='t2')
    parser.add_argument("--training_process", help="forward or backward", type=str, default='forward')
    parser.add_argument("--data_dir", help="data directory", type=str, default='./datasets/data')
    parser.add_argument("--experiment_name", help="model saving file name", type=str, default='None')
    parser.add_argument("--model_name", help="translated model: unet or diffusion", type=str, default='unet')
    args = parser.parse_args()
    main(args)


