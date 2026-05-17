import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb
from pathlib import Path

from aeiou.viz import audio_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..models.wrapper_base import DiffusionModelWrapper, ConditionedDiffusionModelWrapper
from .losses import MSELoss, MultiLoss, ValueLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config

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
#===================================================
class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
            mmvae_conditioning=False
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output", 
                   "targets", 
                   weight=1.0, 
                   mask_key="padding_mask" if self.mask_padding else None, 
                   name="mse_loss"
            )
        ]


        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        self.mmvae_conditioning=mmvae_conditioning

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        # Call at the first time only

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        # ========================================Step0. Conditioning
        #with torch.cuda.amp.autocast():
        with torch.amp.autocast(device_type="cuda"):
            conditioning = self.diffusion.conditioner(metadata, self.device)
            
        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # Create batch tensor of attention masks from the "mask" field of the metadata array
        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        p.tick("conditioning")

        # ========================================Step1. Pretransform
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if self.mmvae_conditioning:
                with torch.amp.autocast(device_type="cuda") and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input, conditioning['prompt'])
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    if use_padding_mask:
                        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
                
            elif not self.pre_encoded:
                #with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                with torch.amp.autocast(device_type="cuda") and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    if use_padding_mask:
                        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        # ========================================Step2. Noise Schedule
        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
            
        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        #with torch.cuda.amp.autocast():
        with torch.amp.autocast(device_type="cuda"):
            p.tick("amp")
            # ========================================Step3. Get Output
            diffusion_output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, return_info=False, **extra_args)
            p.tick("diffusion")
            
            if isinstance(diffusion_output, tuple):
                output, info = diffusion_output
                loss_info.update({
                    "output": output,
                    "targets": targets,
                    "padding_mask": padding_masks if use_padding_mask else None
                })
            else:
                output = diffusion_output
                loss_info.update({
                    "output": output,
                    "targets": targets,
                    "padding_mask": padding_masks if use_padding_mask else None,
                })

            # ========================================Step4. Calculate Loss
            loss, losses = self.losses(loss_info)

            p.tick("loss")

            if self.log_loss_info:
                # Loss debugging logs
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(output, targets, reduction="none")

                sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze() 

                # gather loss_all across all GPUs
                loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

                # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
                loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

                # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
                debug_log_dict = {
                    f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
                }

                self.log_dict(debug_log_dict)


        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        # NOT CALLED
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8, #***
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {}, # ***
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7], #***
                 demo_cond_from_batch: bool = False,
                 display_audio_cond: bool = False,
                 demo_dir=None,
                 wandb_skip=False,
                mmvae_conditioning=False
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales


        if len(demo_conditioning) != num_demos:
            raise Exception("[Error] 'num_demos' must has the same number of samples in demo_cond")

        # If true, the callback will use the metadata from the batch to generate the demo conditioning
        self.demo_cond_from_batch = demo_cond_from_batch

        # If true, the callback will display the audio conditioning
        self.display_audio_cond = display_audio_cond
        self.demo_dir=demo_dir
        self.wandb_skip=wandb_skip
        self.mmvae_conditioning=mmvae_conditioning

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):        
        if self.num_demos==0:
            return


        if self.demo_dir ==None:
            Path('./output_demo').mkdir(parents=True, exist_ok=True)
        else:
            Path(self.demo_dir).mkdir(parents=True, exist_ok=True)

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_samples = self.demo_samples

        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            # Get metadata from the batch
            demo_cond = batch[1][:self.num_demos]

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio


        if not self.mmvae_conditioning:
            noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

            

        try:
            print("Getting conditioning")
            #with torch.cuda.amp.autocast():
            with torch.amp.autocast(device_type="cuda"):
                conditioning = module.diffusion.conditioner(demo_cond, module.device)

            if self.mmvae_conditioning:
                noise = module.diffusion.pretransform.encode_text(conditioning['prompt'], sigma_max=1.0, dist_shift=None)

            # Check this !!
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            log_dict = {}

            if self.display_audio_cond:
                audio_inputs = torch.cat([cond["audio"] for cond in demo_cond], dim=0)
                audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')

                filename = f'demo_audio_cond_{trainer.global_step:08}.wav'
                audio_inputs = audio_inputs.to(torch.float32).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio_inputs, self.sample_rate)
                log_dict[f'demo_audio_cond'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption="Audio conditioning")
                log_dict[f"demo_audio_cond_melspec_left"] = wandb.Image(audio_spectrogram_image(audio_inputs))
                if not self.wandb_skip:
                    trainer.logger.experiment.log(log_dict)

            for cfg_scale in self.demo_cfg_scales:

                print(f"Generating demo for cfg scale {cfg_scale}")
                
                #with torch.cuda.amp.autocast():
                with torch.amp.autocast(device_type="cuda"):
                    model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model

                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    
                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                log_dict = {}
                
                if self.demo_dir ==None:
                    filename = f'output_demo/demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                else:
                    filename = f'{self.demo_dir}/demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                # Cut audio to seconds_total if specified in demo_conditioning
                if self.demo_conditioning and len(self.demo_conditioning) > 0:
                    # fakes: [num_demos, 2, N]
                    for i, cond in enumerate(self.demo_conditioning):
                        seconds_total = cond.get('seconds_total')
                        if seconds_total:
                            target_length = int(seconds_total * self.sample_rate)
                            if fakes.shape[2] > target_length:
                                trimmed = fakes[i, :, :target_length]
                            else:
                                trimmed = fakes[i]
                        else:
                            trimmed = fakes[i]                        
                        # Save individual audio file
                        individual_filename = f'{self.demo_dir if self.demo_dir else "output_demo"}/demo_cfg_{cfg_scale}_{trainer.global_step:08}_sample_{i}.wav'
                        trimmed = trimmed.div(torch.max(torch.abs(trimmed))).mul(32767).to(torch.int16).cpu()
                        torchaudio.save(individual_filename, trimmed, self.sample_rate)

                        log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(individual_filename,
                                                            sample_rate=self.sample_rate,
                                                            caption=f'Reconstructed')
            
                        log_dict[f'demo_melspec_left_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(trimmed))
                        del trimmed
                if not self.wandb_skip:
                    trainer.logger.experiment.log(log_dict)
            
            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()

#===================================================
class DiffusionUncondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training an unconditional audio diffusion model (like Dance Diffusion).
    '''
    def __init__(
            self,
            model: DiffusionModelWrapper,
            lr: float = 1e-4,
            pre_encoded: bool = False,
            use_ema=False
    ):
        super().__init__()

        self.diffusion = model
        
        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                     "targets",
                     weight=1.0,
                     name="mse_loss"
                )
        ]

        self.losses = MultiLoss(loss_modules)

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        
        diffusion_input = reals

        loss_info = {}

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        if self.diffusion.pretransform is not None:
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        loss_info["reals"] = diffusion_input

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas

        #with torch.cuda.amp.autocast():
        with torch.amp.autocast(device_type="cuda"):
            v = self.diffusion(noised_inputs, t)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):

        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionUncondDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 demo_steps=250,
                 sample_rate=48000
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        demo_samples = module.diffusion.sample_size

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            #with torch.cuda.amp.autocast():
            with torch.amp.autocast(device_type="cuda"):

                model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model

                fakes = sample(model, noise, self.demo_steps, 0)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        
            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            trainer.logger.experiment.log(log_dict)

            del fakes
            
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
        finally:
            gc.collect()
            torch.cuda.empty_cache()


#===================================================
