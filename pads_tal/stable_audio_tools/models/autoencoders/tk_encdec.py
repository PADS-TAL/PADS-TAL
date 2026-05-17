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

        self.init_conv = WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3)
        
        self.decoder_blocks = nn.ModuleList()
        self.res_h = nn.ModuleList()
        self.res_z_fc = nn.ModuleList()
        self.res_z_conv = nn.ModuleList()
        self.activation_fn = nn.Sigmoid()
           
        for i in range(self.depth-1, 0, -1):
            in_ch = c_mults[i] * channels
            out_ch = c_mults[i-1] * channels

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    stride=strides[i-1], 
                    use_snake=use_snake, 
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample
                )
            )
            self.res_h.append(nn.Conv1d(out_ch, out_ch, kernel_size=1))
            self.res_z_fc.append(nn.Linear(3875, 31000)) # TODO 3875
            self.res_z_conv.append(nn.Conv1d(64, out_ch, kernel_size=1))

        self.final_act = get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
        self.final_conv = WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
        self.output_nonlinear = nn.Tanh() if final_tanh else nn.Identity()


    def forward(self, x):
        residual = x 

        x = self.init_conv(x) 

        for i, block in enumerate(self.decoder_blocks):
            x = block(x) 
            x_proj = self.res_h[i](x)
            h_proj = self.res_z
            h_proj = self.res_z[i](residual)
            x = self.activation_fn(x_proj + h_proj)

        x = self.final_act(x)
        x = self.final_conv(x)
        return self.output_nonlinearity(x)


    
# AE factories
def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
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
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

