import torch
from einops import rearrange
from torch import nn

class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def decode_tokens(self, tokens):
        raise NotImplementedError


#=============================================
class MultiModalVAEPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False):
        super().__init__(enable_grad=False, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch
        

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x_audio, x_text, **kwargs):
        
        if isinstance(x_text, list) and isinstance(x_text[0], str):
            text_attention_mask = None
        else:
            x_text, text_attention_mask = x_text
        if self.model_half:
            x_audio = x_audio.half()
            x_text = x_text.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_text_audio(x_text, x_audio, chunked=self.chunked, iterate_batch=self.iterate_batch, text_attention_mask=text_attention_mask, **kwargs)

        if self.model_half:
            encoded = encoded.float()
    
        return encoded / self.scale

    def encode_text(self, x_text, **kwargs):
        if isinstance(x_text, list) and isinstance(x_text[0], str):
            text_attention_mask = None
        else:
            x_text, text_attention_mask = x_text
        
        if self.model_half:
            x_text = x_text.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_text(x_text, chunked=self.chunked, iterate_batch=self.iterate_batch, text_attention_mask=text_attention_mask, **kwargs)

        if self.model_half:
            encoded = encoded.float()
    
        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False):
        super().__init__(enable_grad=False, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch
        

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()
    
        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

#=================================================================
def create_pretransform_from_config(pretransform_config, sample_rate):

    #====================================================Simple
    pretransform_type = pretransform_config.get('type', None)
    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == 'autoencoder':
        #====================================================Simple
        # ***********************************************
        from .model_autoencoders import create_autoencoder_from_config
        #from .model_pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = AutoencoderPretransform(autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked)
    elif pretransform_type == 'autoencoder_mvae':
        #====================================================Simple
        # ***********************************************
        from .model_mvae import create_mvae_from_config
        #from .model_pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_mvae_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = MultiModalVAEPretransform(autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked)
        #====================================================END
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')
    
    enable_grad = pretransform_config.get('enable_grad', False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform

