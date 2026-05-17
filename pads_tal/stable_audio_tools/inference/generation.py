import numpy as np
import torch 
import typing as tp
import math 

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf


#conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size
def generate_diffusion_cond(
        model,

        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",

        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        mask_args: dict = None, # OLD Arguments
        return_latents = False,

        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size
    diff_objective = model.diffusion_objective

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed

    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)


    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # **********************************Step0. Conditioning
    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    # **********************************Step1. initialize audio if exists
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        if len(init_audio.shape)==2:
            # Prepare the initial audio for use by the model 
            init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

            # For latent models, encode the initial audio into latents
            if model.pretransform is not None:
                init_audio = model.pretransform.encode(init_audio)

            init_audio = init_audio.repeat(batch_size, 1, 1)
        elif len(init_audio.shape)==3:
            init_audio_out = []
            for init_audio_single in init_audio :
                # Prepare the initial audio for use by the model 
                init_audio_single = prepare_audio(init_audio_single, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

                # For latent models, encode the initial audio into latents
                if model.pretransform is not None:
                    init_audio_single = model.pretransform.encode(init_audio_single)

                init_audio_out.append(init_audio_single)
            init_audio = torch.cat(init_audio_out, dim=0)
        else: 
            raise Exception("[SAAudio][Error] Werid init audio shape : {}".format(init_audio.shape))
            
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None
        mask_args = None

    # **********************************Step2. masking for inpainting if exists
    # Inpainting mask
    if init_audio is not None and mask_args is not None:
        # Cut and paste init_audio according to cropfrom, pastefrom, pasteto
        # This is helpful for forward and reverse outpainting
        cropfrom = math.floor(mask_args["cropfrom"]/100.0 * sample_size)
        pastefrom = math.floor(mask_args["pastefrom"]/100.0 * sample_size)
        pasteto = math.ceil(mask_args["pasteto"]/100.0 * sample_size)
        assert pastefrom < pasteto, "Paste From should be less than Paste To"

        croplen = pasteto - pastefrom
        if cropfrom + croplen > sample_size:
            croplen = sample_size - cropfrom 
        cropto = cropfrom + croplen

        pasteto = pastefrom + croplen
        cutpaste = init_audio.new_zeros(init_audio.shape)
        cutpaste[:, :, pastefrom:pasteto] = init_audio[:,:,cropfrom:cropto]

        init_audio = cutpaste

        # Build a soft mask (list of floats 0 to 1, the size of the latent) from the given args

        if model.pretransform is not None:
            mask = build_mask(sample_size, mask_args, model.pretransform.downsampling_ratio)
        else:
            mask = build_mask(sample_size, mask_args)
        mask = mask.to(device)
    elif init_audio is not None and mask_args is None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None and isinstance(v, torch.Tensor) else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!


    # **********************************Step3. Sampling
    if diff_objective == "v":    
        # k-diffusion denoising process go!
        
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        #sampled = sample(model.model, noise, steps, 0, **conditioning_inputs, cfg_scale=cfg_scale, batch_cfg=True)

    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "sampler_type" in sampler_kwargs:
            del sampler_kwargs["sampler_type"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio

    # **********************************Step4. Decoding
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)

        sampled = model.pretransform.decode(sampled)
    # Return audio
    return sampled

# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args, downsampling_ratio=None):
    maskstart = mask_args["maskstart"]
    maskend = mask_args["maskend"]
    softnessL = mask_args["softnessL"]
    softnessR = mask_args["softnessR"]
    if downsampling_ratio != None:
        maskstart = maskstart // downsampling_ratio
        maskend = maskend // downsampling_ratio
    if softnessL==0 and softnessR==0:
        softness=0
    elif downsampling_ratio != None :
        softness = round((maskend-maskstart)*0.1)
    
    marination = mask_args["marination"]

    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softness*2, periodic=False)[:softness]
    hannR = torch.hann_window(softness*2, periodic=False)[softness:]

    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softness] = hannL
    mask[maskend-softness:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    return mask

