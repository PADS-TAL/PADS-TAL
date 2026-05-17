import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import numpy as np
import typing as tp
import math

from .common.blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .conditioners import create_multi_conditioner_from_conditioning_config
from .diffusions.dit import DiffusionTransformer
from .model_pretransforms import create_pretransform_from_config

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep



class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

# ************************************************************
class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = DiffusionTransformer(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            input_concat_cond=input_concat_cond,
            global_embed=global_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,

            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,

            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            **kwargs)

def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    # ==========================================Simple
    model_type = config["model_type"]

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"


    # ==========================================Simple

    model_config = config["model"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    #*********pretransform
    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    #*********conditioner
    conditioning_config = model_config.get('conditioning', None)
    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config, pretransform=pretransform)

    possible_conditions = [c['id'] for c in conditioning_config.get('configs',[])]
    send_prompt = conditioning_config.get('send_prompt', False)

    #*********diffusion

    # Get Real ModelWrapper Type
    if diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config) #****************************
    else:
        raise Exception("[Error] Not supported diffusion model type : {}".format(diffusion_model_type))

    if diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v') #*********
    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])

    given_conditions = cross_attention_ids+global_cond_ids+input_concat_ids+prepend_cond_ids
    given_conditions = list(set(given_conditions))
    if len(possible_conditions)==len(given_conditions):
        print("\n[Info] Exactly Matched Conditions\n")
    else:
        print("\n[Warning] Possible Conditions({}) not match with given Conditions({})\n".format(possible_conditions, given_conditions))

    #*********etc
    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"
    # ==========================================End
    # Get AllWrappper class
    extra_kwargs = {}
    if model_type == "diffusion_cond" :
        from .wrapper_base import ConditionedDiffusionModelWrapper
        wrapper_fn = ConditionedDiffusionModelWrapper
        extra_kwargs["diffusion_objective"] = diffusion_objective

        return wrapper_fn(
            diffusion_model,
            conditioner,
            min_input_length=min_input_length,
            sample_rate=sample_rate,
            cross_attn_cond_ids=cross_attention_ids,
            global_cond_ids=global_cond_ids,
            input_concat_ids=input_concat_ids,
            prepend_cond_ids=prepend_cond_ids,
            pretransform=pretransform,
            io_channels=io_channels,
            send_prompt=send_prompt,
            **extra_kwargs
        )
