import ml_collections

import torch as th

def get_default_configs():
    config = ml_collections.ConfigDict()
    config.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    config.seed = 1
    config.data = data = ml_collections.ConfigDict()
    data.path = "/home/some5338/Documents/data/brats"
    data.sequence_translation = False # bool
    data.healthy_data_percentage = None

    ## Diffusion parameters
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.steps = 1000
    diffusion.learn_sigma = False
    diffusion.sigma_small = False
    diffusion.noise_schedule = "linear" #linear, cosine
    diffusion.predict_xstart = True   # if True, then predict xstart each iteration, else predict noise (for cold diffusion, set this to true !)
    diffusion.rescale_timesteps = False
    diffusion.conditioning_noise = "constant"  # "constant" or "reverse"

    ## score model config
    config.score_model = score_model = ml_collections.ConfigDict()
    score_model.image_size = 128
    score_model.num_input_channels = 1
    score_model.num_channels = 32 #64,96
    score_model.num_res_blocks = 2 #2,3
    score_model.num_heads = 1
    score_model.num_heads_upsample = -1
    score_model.num_head_channels = -1
    score_model.learn_sigma = diffusion.learn_sigma
    score_model.attention_resolutions = "32,16,8"  # 16

    attention_ds = []
    if score_model.attention_resolutions != "":
        for res in score_model.attention_resolutions.split(","):
            attention_ds.append(score_model.image_size // int(res))
    score_model.attention_ds = attention_ds

    score_model.channel_mult = {64:(1, 2, 3, 4), 128:(1, 1, 2, 3, 4)}[score_model.image_size]
    score_model.dropout = 0.1
    score_model.use_checkpoint = False
    score_model.use_scale_shift_norm = True
    score_model.resblock_updown = True
    score_model.use_fp16 = False
    score_model.use_new_attention_order = False

    # score model training
    config.score_model.training = training_score = ml_collections.ConfigDict()
    training_score.schedule_sampler = "uniform"  # "uniform" or "loss-second-moment"
    training_score.lr = 1e-4
    training_score.weight_decay = 0.00
    training_score.lr_decay_steps = 150000
    training_score.lr_decay_factor = 0.5
    training_score.batch_size = 32
    training_score.ema_rate = "0.9999"  # comma-separated list of EMA values
    training_score.log_interval = 100
    training_score.save_interval = 5000
    training_score.use_fp16 = score_model.use_fp16
    training_score.fp16_scale_growth = 1e-3
    training_score.iterations = 80000

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.clip_denoised = False
    sampling.dynamic_sampling = True
    sampling.num_samples = 125      # 1024, 3z986
    sampling.batch_size = 32
    sampling.ivlr_scale = 16
    sampling.reconstruction = True
    sampling.eta = 0.0
    sampling.label_of_intervention = "gt"
    sampling.classifier_scale = 1
    sampling.norm_cond_scale = 3.0
    sampling.sampling_progression_ratio = 0.8  ### originally 0.75
    sampling.sdedit_inpaint = False
    sampling.detection = True
    sampling.source_class = 1 # int in range [0, num_class-1]
    sampling.target_class = 1 # int in range [0, num_class-1]
    sampling.progress = True

    return config

