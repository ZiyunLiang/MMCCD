import ml_collections

from models import gaussian_diffusion as gd
from models import unet
from models.respace import SpacedDiffusion, space_timesteps


def create_score_model(config: ml_collections.ConfigDict, image_level_cond):
    return unet.UNetModel(
        in_channels=config.score_model.num_input_channels,
        model_channels=config.score_model.num_channels,
        out_channels=(
            config.score_model.num_input_channels
            if not config.score_model.learn_sigma else 2 * config.score_model.num_input_channels),
        num_res_blocks=config.score_model.num_res_blocks,
        attention_resolutions=tuple(config.score_model.attention_ds),
        dropout=config.score_model.dropout,
        channel_mult=config.score_model.channel_mult,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=config.score_model.num_heads,
        num_head_channels=config.score_model.num_head_channels,
        num_heads_upsample=config.score_model.num_heads_upsample,
        use_scale_shift_norm=config.score_model.use_scale_shift_norm,
        resblock_updown=config.score_model.resblock_updown,
        image_level_cond=image_level_cond,
    )



def create_gaussian_diffusion(config, timestep_respacing):
    betas = gd.get_named_beta_schedule(config.diffusion.noise_schedule, config.diffusion.steps)

    if not timestep_respacing:
        timestep_respacing = config.diffusion.steps
    return SpacedDiffusion(
        use_timesteps=space_timesteps(config.diffusion.steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not config.diffusion.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not config.diffusion.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not config.diffusion.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        rescale_timesteps=config.diffusion.rescale_timesteps,
        conditioning_noise=config.diffusion.conditioning_noise
    )


