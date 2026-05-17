import torch

from torch import nn
from torchaudio import transforms as T
from typing import Dict, Any
from einops import rearrange

from ..inference.utils import prepare_audio
from .model_bottleneck import Bottleneck, DiscreteBottleneck
from .autoencoders.encdec import create_encoder_from_config, create_decoder_from_config
from .autoencoders.textenc import T5Encoder, TextDecoder
from .model_bottleneck import create_bottleneck_from_config

from einops import rearrange

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

# ==================================================
class MultiModalVAE(nn.Module):
    def __init__(
        self,
        encoder_audio,
        encoder_text,
        decoder_audio,
        decoder_text,
        private_audio,
        private_text,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False,
        private_weaker=False,
        private_audio_latents=128,
        alternative_encoders=False,
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.private_weaker = private_weaker

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck
        if self.bottleneck ==None:
            raise Exception("[Error] encoder must be made, but None")

        self.encoder_audio = encoder_audio
        #if self.encoder_audio ==None:
        #    raise Exception("[Error] encoder must be made, but None")

        self.encoder_text = encoder_text
        if self.encoder_text ==None:
            raise Exception("[Error] encoder must be made, but None")
        self.decoder_audio = decoder_audio
        if self.decoder_audio ==None:
            raise Exception("[Error] encoder must be made, but None")
        self.decoder_text = decoder_text
        #if self.decoder_text ==None:
        #    raise Exception("[Error] encoder must be made, but None")

        self.private_audio = private_audio
        if self.private_audio ==None:
            raise Exception("[Error] encoder must be made, but None")
        self.private_audio_latents = private_audio_latents
        self.private_text = private_text
        #if self.private_text ==None:
        #    raise Exception("[Error] encoder must be made, but None")

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck.is_discrete 



        self.alternative_encoders = alternative_encoders
            

    def encode(self, text, audio=None, return_all=False, return_info=False, iterate_batch=False, text_attention_mask=None, sigma_max=None, dist_shift=None, diff_objective="v", **kwargs):

        if (self.encoder_audio==None or self.private_text==None) and not self.alternative_encoders:
            return_all=False
            
        info = {}

        # ============================================1. Shared Text Encoder
        if iterate_batch:
            latents_text = []
            if isinstance(text, torch.Tensor):
                for i in range(text.size(0)):
                    tt = self.encoder_text(text[i:i+1], audio.device if audio is not None else "cuda", attention_mask=text_attention_mask[i:i+1])
                    latents_text.append(tt)
            else :
                ids = []
                for i in range(len(text)):
                    tt, tids = self.encoder_text(text[i:i+1], audio.device if audio is not None else "cuda", return_ids=True)
                    latents_text.append(tt)
                    ids.append(tids)
                ids = torch.cat(ids, dim=0)
                info["reals_text_ids"] = ids 
            latents_text = torch.cat(latents_text, dim=0)
        else:
            if isinstance(text, torch.Tensor):
                latents_text = self.encoder_text(text,  audio.device if audio is not None else "cuda", attention_mask=text_attention_mask[i:i+1])
            else:
                latents_text, ids = self.encoder_text(text,  audio.device if audio is not None else "cuda", return_ids=True)
                info["reals_text_ids"] = ids 

        shared_text_latents, bottleneck_info = self.bottleneck.encode(latents_text, return_info=True, kl_name="kl_shared_text", chunk_dim=-1,info_getmean="shared_latents_text_mean", **kwargs)
        info.update(bottleneck_info)

        if audio==None :
            B, shared_ch, D = shared_text_latents.shape      # shared_ch = 32, D = 1024
            private_ch = self.private_audio_latents          # private_ch = 64

            full_noise = torch.randn(
                    (B, shared_ch + private_ch, D),
                    dtype=shared_text_latents.dtype,
                    device=shared_text_latents.device)

            shared_text_noise   = full_noise[:, :shared_ch, :]         # [B, 32, 1024]
            private_audio_noise = full_noise[:, shared_ch:, :]         # [B, 64, 1024]

            if diff_objective=="v" : # sample_k
                if sigma_max == None :
                    shared_text_latents = shared_text_latents 
                else:
                    shared_text_latents = shared_text_latents + shared_text_noise *sigma_max 
            elif diff_objective=="rectified_flow": # sample_rf
                if sigma_max == None :
                    raise Exception("[Error] Sigma max must be given for mvae text encoding")
                sigma_max = dist_shift.time_shift(torch.tensor(sigma_max), shared_text_latents.shape[-1]).item()

                shared_text_latents = shared_text_latents * (1-sigma_max) + shared_text_noise * sigmax_max 
            else: # sample_k - k-diff
                raise Exception("[Error] Unsupported diff objective : {}".format(diff_objective))
                
            text_audio =torch.cat([shared_text_latents, private_audio_noise], dim=1) # Cross! for Audio

            if return_info:
                return text_audio, info
            else:
                return text_audio

        # ============================================2. Private Audio Encoder
        if iterate_batch:
            private_latents_audio = []
            for i in range(audio.shape[0]):
                private_latents_audio.append(self.private_audio(audio[i:i+1]))
            private_latents_audio = torch.cat(private_latents_audio, dim=0)
        else:
            private_latents_audio = self.private_audio(audio)

        if not self.alternative_encoders:
            private_latents_audio, bottleneck_info = self.bottleneck.encode(private_latents_audio, return_info=True, kl_name="kl_audio", **kwargs)
            info.update(bottleneck_info)
        else:
            private_latents_audio, bottleneck_info = self.bottleneck.encode(private_latents_audio, return_info=True, kl_name="kl_audio", info_getmean='shared_latents_audio_mean', **kwargs)
            info.update(bottleneck_info)
            
        if self.private_weaker :
            text_audio =torch.cat([shared_text_latents, torch.randn_like(private_latents_audio)], dim=1) # Cross! for Audio
        else:
            text_audio =torch.cat([shared_text_latents, private_latents_audio], dim=1) # Cross! for Audio

        if not return_all:
            if return_info:
                return text_audio, info
            return text_audio


        # ============================================3. Shared Audio Encoder
        if not self.alternative_encoders:
            if iterate_batch:
                latents_audio = []
                for i in range(audio.shape[0]):
                    latents_audio.append(self.encoder_audio(audio[i:i+1]))
                latents_audio = torch.cat(latents_audio, dim=0)
            else:
                latents_audio = self.encoder_audio(audio)

            shared_audio_latents, bottleneck_info = self.bottleneck.encode(latents_audio, return_info=True, kl_name="kl_shared_audio",info_getmean="shared_latents_audio_mean", **kwargs)
            info.update(bottleneck_info)
            #info["shared_latents_audio"] = shared_audio_latents
        else:
            shared_audio_latents = private_latents_audio
            info.update({'kl_shared_audio':info['kl_audio']})

        audio_audio =torch.cat([shared_audio_latents, private_latents_audio], dim=1)  # Self! for Audio
        
        # ============================================4. Private Text Encoder
        if not self.alternative_encoders:
            if iterate_batch:
                private_latents_text = []
                if isinstance(text, torch.Tensor):
                    for i in range(text.size(0)):
                        private_latents_text.append(self.private_text(text[i:i+1],audio.device, attention=text_attention_mask[i:i+1]))
                else:
                    for i in range(len(text)):
                        private_latents_text.append(self.private_text(text[i:i+1],audio.device))
                private_latents_text = torch.cat(private_latents_text, dim=0)
            else:
                if isinstance(text, torch.Tensor):
                    private_latents_text = self.private_text(text, audio.device, attention=text_attention_mask[i:i+1])
                else:
                    private_latents_text = self.private_text(text, audio.device)
            
            private_latents_text, bottleneck_info = self.bottleneck.encode(private_latents_text, return_info=True, kl_name="kl_text", chunk_dim=-1, **kwargs)
            info.update(bottleneck_info)
        else:
            private_latents_text = shared_text_latents
            info.update({'kl_text':info['kl_shared_text']})


        if self.private_weaker :
            audio_text =torch.cat([shared_audio_latents, torch.randn_like(private_latents_text)], dim=-1) # Cross! For Text
        else:
            audio_text =torch.cat([shared_audio_latents, private_latents_text], dim=-1) # Cross! For Text

        text_text =torch.cat([shared_text_latents, private_latents_text], dim=-1) # Self! For text

        if return_info:
            return text_audio, audio_audio, audio_text, text_text, info
        return text_audio, audio_audio, audio_text, text_text

    def decode(self, latents_text_audio, latents_audio_audio=None, latents_audio_text=None, latents_text_text=None, iterate_batch=False, decoder_text_freeze=False, **kwargs):


        if self.decoder_text==None:
            latents_audio_text = None
            latents_text_text = None
            

        # ============================================Audio Decoder : Shared Text + Private Audio
        if iterate_batch:
            decoded_audio_from_text = []
            for i in range(latents_text_audio.shape[0]):
                decoded_audio_from_text.append(self.decoder_audio(latents_text_audio[i:i+1]))
            decoded_audio_from_text = torch.cat(decoded_audio_from_text, dim=0)
        else:
            decoded_audio_from_text = self.decoder_audio(latents_text_audio, **kwargs)

        if self.soft_clip:
            decoded_audio_from_text = torch.tanh(decoded_audio_from_text)

        if latents_audio_audio==None:
            return decoded_audio_from_text 

        # ============================================Audio Decoder : Shared Audio + Private Audio
        if iterate_batch:
            decoded_audio_from_audio = []
            for i in range(latents_audio_audio.shape[0]):
                decoded_audio_from_audio.append(self.decoder_audio(latents_audio_audio[i:i+1]))
            decoded_audio_from_audio = torch.cat(decoded_audio_from_audio, dim=0)
        else:
            decoded_audio_from_audio = self.decoder_audio(latents_audio_audio, **kwargs)

        if self.soft_clip:
            decoded_audio_from_audio = torch.tanh(decoded_audio_from_audio)

        if latents_audio_text==None or latents_text_text==None:
            return decoded_audio_from_text, decoded_audio_from_audio
        
        # ============================================Text Decoder : Shared Audio + Private Text
        if iterate_batch :
            if decoder_text_freeze:
                with torch.no_grad():
                    decoded_text_from_audio = []
                    for i in range(latents_audio_text.shape[0]):
                        decoded_text_from_audio.append(self.decoder_text(latents_audio_text[i:i+1]))
                    decoded_text_from_audio = torch.cat(decoded_text_from_audio, dim=0)
            else:
                decoded_text_from_audio = []
                for i in range(latents_audio_text.shape[0]):
                    decoded_text_from_audio.append(self.decoder_text(latents_audio_text[i:i+1]))
                decoded_text_from_audio = torch.cat(decoded_text_from_audio, dim=0)
        else:
            if decoder_text_freeze:
                with torch.no_grad():
                    decoded_text_from_audio = self.decoder_text(latents_audio_text, **kwargs) 
            else:
                decoded_text_from_audio = self.decoder_text(latents_audio_text, **kwargs) 

        if self.soft_clip:
            decoded_text_from_audio = torch.tanh(decoded_text_from_audio)

        # ============================================Text Decoder : Shared Text + Private Text
        if iterate_batch :
            if decoder_text_freeze:
                with torch.no_grad():
                    decoded_text_from_text = []
                    for i in range(latents_text_text.shape[0]):
                        decoded_text_from_text.append(self.decoder_text(latents_text_text[i:i+1]))
                    decoded_text_from_text = torch.cat(decoded_text_from_text, dim=0)
            else:
                decoded_text_from_text = []
                for i in range(latents_text_text.shape[0]):
                    decoded_text_from_text.append(self.decoder_text(latents_text_text[i:i+1]))
                decoded_text_from_text = torch.cat(decoded_text_from_text, dim=0)
        else:
            if decoder_text_freeze:
                with torch.no_grad():
                    decoded_text_from_text = self.decoder_text(latents_text_text, **kwargs) 
            else:
                decoded_text_from_text = self.decoder_text(latents_text_text, **kwargs) 

        if self.soft_clip:
            decoded_text_from_text = torch.tanh(decoded_text_from_text)

        return decoded_audio_from_text, decoded_audio_from_audio, decoded_text_from_audio, decoded_text_from_text


          
    # *************TOOLS 1
    def decode_tokens(self, tokens, **kwargs):
        raise Exception("Not Implemented because this has continuous bottleneck")
        
    def encode_text_audio(self, text, audio, overlap=32, chunk_size=128, text_attention_mask=None, **kwargs):
        return self.encode(text, audio=audio, text_attention_mask=text_attention_mask, **kwargs)

    def encode_text(self, text, overlap=32, chunk_size=128, text_attention_mask=None, **kwargs):

        return self.encode(text, text_attention_mask=text_attention_mask, **kwargs)
    
    def decode_audio(self, latents, overlap=32, chunk_size=128, **kwargs):

        model_dtype = next(self.parameters()).dtype
        latents = latents.to(model_dtype)

        # default behavior. Decode the entire latent in parallel
        return self.decode(latents, **kwargs)

    
#==============================================================!!
def create_mvae_from_config(config: Dict[str, Any]):
    
    # ==========================================Simple
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"
    # ==========================================Simple
    ae_config = config["model"]

    alternative_encoders = ae_config.get("alternative_encoders", False)  

    #*********encoder & decoder
    encoder_audio_config = ae_config.get("encoder_audio", None) 
    if encoder_audio_config and not alternative_encoders :
        encoder_audio = create_encoder_from_config(encoder_audio_config)
    else:
        # alternative : encoder_audio <---- private_audio
        encoder_audio = None
    decoder_audio = create_decoder_from_config(ae_config["decoder_audio"])

    encoder_text = T5Encoder(**ae_config["encoder_text"]["config"])
    decoder_text_config = ae_config.get("decoder_text", None) 
    if decoder_text_config:
        decoder_text = TextDecoder(encoder_text.tokenizer, **decoder_text_config["config"])
    else:
        decoder_text = None

    private_audio_config = ae_config["private_audio"]
    private_audio = create_encoder_from_config(private_audio_config)
    private_audio_latents = private_audio_config.get("latent_dim", 128)//2
    private_text_config = ae_config.get("private_text", None) 
    if private_text_config and not alternative_encoders:
        # alternative : encoder_audio <---- private_audio
        private_text = T5Encoder(**private_text_config["config"])
    else:
        private_text = None


    #*********bottleneck
    bottleneck = ae_config.get("bottleneck", None)
    if bottleneck is None:
        raise Exception("mvae must need bottleneck")
    bottleneck = create_bottleneck_from_config(bottleneck)


    # ==========================================Simple
    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"

    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"

    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)
    soft_clip = ae_config["decoder_audio"].get("soft_clip", False)
    private_weaker = ae_config.get("private_weaker", False)


    # ==========================================Simple
    return MultiModalVAE(
        encoder_audio,
        encoder_text,
        decoder_audio,
        decoder_text,
        private_audio,
        private_text,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip,
        private_weaker=private_weaker,
        private_audio_latents=private_audio_latents,
        alternative_encoders=alternative_encoders,
    )

