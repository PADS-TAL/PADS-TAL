import torch
import math

from torch import nn
from torch.nn.utils import weight_norm
from alias_free_torch import Activation1d
#from dac.nn.layers import WNConv1d, WNConvTranspose1d
from typing import Literal, Dict, Any

from ..common.blocks import SnakeBeta


from ..common.transformer import TransformerBlock


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act



class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        
        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        res = x
        
        x = checkpoint(self.layers, x)
        #x = self.layers(x)

        return x + res


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.transpose(-1,-2)

class TAAEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, type = 'encoder', transformer_depth = 3, use_snake = False, sliding_window = [31,32], checkpointing = False, conformer = False, layer_scale = True, use_dilated_conv = False):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")
        
        self.checkpointing = checkpointing
        
        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        transformers.append(Transpose())

        for _ in range(transformer_depth):
            transformers.append(TransformerBlock(transformer_dim, 
                                                 dim_heads = 128, 
                                                 causal = False, 
                                                 zero_init_branch_outputs = True if not layer_scale else False, 
                                                 remove_norms = False, 
                                                 conformer = conformer, 
                                                 layer_scale = layer_scale, 
                                                 add_rope = True, 
                                                 attn_kwargs={'sliding_window': sliding_window, 'qk_norm': "ln"}, 
                                                 ff_kwargs={'mult': 4, 'no_bias': False},
                                                 norm_kwargs = {'eps': 1e-2}))
        transformers.append(Transpose())
        transformers = nn.Sequential(*transformers)

        if type == 'encoder':
            layers = []
            if stride > 1 or in_channels != out_channels:
                conv_type = WNConv1d
            else:
                conv_type = nn.Identity
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake))
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=in_channels))
            layers.append(conv_type(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)))
            layers.append(transformers)
            self.layers = nn.Sequential(*layers)
        elif type == 'decoder':
            layers = []
            if stride > 1 or in_channels != out_channels:
                conv_type = WNConvTranspose1d
            else:
                conv_type = nn.Identity
            layers.append(transformers)
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=in_channels))
            layers.append(conv_type(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)))
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake))
            self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self.layers, x)
        else:
            return self.layers(x)

class TAAEEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()
          
        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [WNConv1d(in_channels=in_channels, out_channels=channel_dims[0], kernel_size=7, padding=3, bias = True)]

        for i in range(self.depth):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]

        layers += [
            get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[-1]),
            WNConv1d(in_channels=channel_dims[-1], out_channels=latent_dim, kernel_size=3, padding=1, bias = True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TAAEDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=channel_dims[-1], kernel_size=3, padding=1, bias = True)
        ]
        
        for i in range(self.depth, 0, -1):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]  

        layers += [get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[0]),
                    WNConv1d(in_channels=channel_dims[0], out_channels=out_channels, kernel_size=7, padding=3, bias = False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Decoder as DACDecoder

        self.decoder = DACDecoder(**kwargs, input_channel = latent_dim, d_out=out_channels)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)


    
# AE factories
def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"]
        )
    
    elif encoder_type == "seanet":
        from encodec.modules import SEANetEncoder
        seanet_encoder_config = encoder_config["config"]

        #SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(
            **seanet_encoder_config
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]

        encoder = DACEncoderWrapper(**dac_config)
    elif encoder_type == "local_attn":
        from .local_attention import TransformerEncoder1D

        local_attn_config = encoder_config["config"]

        encoder = TransformerEncoder1D(
            **local_attn_config
        )
    elif encoder_type == "taae":
        encoder = TAAEEncoder(
            **encoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "seanet":
        from encodec.modules import SEANetDecoder

        decoder = SEANetDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]

        decoder = DACDecoderWrapper(**dac_config)
    elif decoder_type == "local_attn":
        from .local_attention import TransformerDecoder1D

        local_attn_config = decoder_config["config"]

        decoder = TransformerDecoder1D(
            **local_attn_config
        )
    elif decoder_type == "taae":
        decoder = TAAEDecoder(
            **decoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

