import os
import torch
import torchaudio
import wandb
from einops import rearrange
from safetensors.torch import save_model
from ema_pytorch import EMA
import pytorch_lightning as pl
from pathlib import Path

from copy import deepcopy
from typing import Optional, Literal

from .losses import auraloss as auraloss
from .losses import MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, TargetValueLoss, L1Loss, LossWithTarget, MSELoss, HubertLoss, PESQMetric
from .losses import CrossEntropyTokenLossDouble, LatentAlignmentLoss, AuralossLossDouble, L1LossDouble
from ..models.model_mvae import MultiModalVAE, fold_channels_into_batch, unfold_channels_from_batch
from ..models.common.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator
from ..models.model_bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, logger_project_name

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image

import numpy as np
def log_image_tensorboard(logger, grads, name, step):
    writer = logger.experiment  # SummaryWriter object
    img = tokens_spectrogram_image(grads.mean(dim=1).log10(), title = 'Discriminator Sensitivity', symmetric = False)
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    writer.add_image(name, img, global_step=step)
def print_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
def print_norm2(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return torch.nn.utils.get_total_norm(grads, norm_type=2.0)
def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b
def trim_to_shortest_triple(a, b, c):
    """Trim the longer of two tensors to the length of the shorter one."""
    min_length = min(a.shape[-1], b.shape[-1], c.shape[-1])
    return a[:,:,:min_length], b[:,:,:min_length], c[:,:,:min_length]

class MultiModalVAETrainingWrapper(pl.LightningModule):
    def __init__(
            self,
            autoencoder: MultiModalVAE,
            sample_rate=48000,
            loss_config: Optional[dict] = None,
            optimizer_configs: Optional[dict] = None,
            lr: float = 1e-4,
            warmup_steps: int = 0,
            warmup_mode: Literal["adv", "full"] = "adv",
            encoder_freeze_on_warmup: bool = False,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            clip_grad_norm_gen = 0.0,
            clip_grad_norm_disc = 0.0,
            clip_grad_print :int =0,
            separate_disc:bool=False,
            decoder_text_freeze : bool=False,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.clip_grad_norm_gen = clip_grad_norm_gen
        self.clip_grad_norm_disc = clip_grad_norm_disc
        self.clip_grad_print = clip_grad_print
        self.separate_disc=separate_disc
        self.decoder_text_freeze = decoder_text_freeze

        self.force_input_mono = force_input_mono


        if optimizer_configs is None:
            raise Exception("Optimizer Configs must be given")

        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            raise Exception("Loss Configs must be given")


        self.loss_config = loss_config

        # Spectral reconstruction loss
        stft_loss_args = loss_config['spectral']['config']

        self.use_disc = 'discriminator' in loss_config

        if self.autoencoder.out_channels == 2:
            self.sdstft = auraloss.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            self.lrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:
            self.sdstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        # Discriminator
        if self.use_disc:
            if loss_config['discriminator']['type'] == 'oobleck':
                self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'encodec':
                self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'dac':
                self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'big_vgan':
                self.discriminator = BigVGANDiscriminator(channels=self.autoencoder.out_channels, sample_rate=sample_rate,**loss_config['discriminator']['config'])
            
        else:
            self.discriminator = None

        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        if self.use_disc:
            self.gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
                #ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss'),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching'),
            ]
        stft_loss_decay = self.loss_config['spectral'].get('decay', 1.0)

        # Reconstruction loss ****** 
        self.gen_loss_modules += [
            AuralossLossDouble(self.sdstft, target_key = 'reals_audio', input_key = 'decoded_audio_from_text', input_key2= "decoded_audio_from_audio", name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft'], decay = stft_loss_decay),
        ]

        if self.autoencoder.out_channels == 2:
            # Add left and right channel reconstruction losses in addition to the sum and difference
            self.gen_loss_modules += [
                AuralossLossDouble(self.lrstft, target_key = 'reals_audio_left',  input_key = 'decoded_audio_left_from_text', input_key2="decoded_audio_left_from_audio", name='stft_loss_left', weight=self.loss_config['spectral']['weights']['mrstft']/2, decay = stft_loss_decay),
                AuralossLossDouble(self.lrstft, target_key = 'reals_audio_right', input_key = 'decoded_audio_right_from_text', input_key2="decoded_audio_right_from_audio", name='stft_loss_right', weight=self.loss_config['spectral']['weights']['mrstft']/2, decay = stft_loss_decay),
            ]


        if "l1" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l1'] > 0.0:
                self.gen_loss_modules.append(L1LossDouble(key_a='reals_audio', key_b='decoded_audio_from_text',
                                            key_b2='decoded_audio_from_audio',
                                             weight=self.loss_config['time']['weights']['l1'],
                                             name='l1_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += [ValueLoss(key="kl_shared_audio", weight=loss_config['bottleneck']['weights']['kl_shared_audio'], name='kl_shared_audio')]
            
            self.gen_loss_modules += [ValueLoss(key="kl_shared_text", weight=loss_config['bottleneck']['weights']['kl_shared_text'], name='kl_shared_text')]
            
            self.gen_loss_modules += [ValueLoss(key="kl_audio", weight=loss_config['bottleneck']['weights']['kl_audio'], name='kl_audio')]
            
            self.gen_loss_modules += [ValueLoss(key="kl_text", weight=loss_config['bottleneck']['weights']['kl_text'], name='kl_text')]


        if not self.decoder_text_freeze:
            self.gen_loss_modules += [CrossEntropyTokenLossDouble(
                logits_key='decoded_text_logits_from_audio',
                logits_key2='decoded_text_logits_from_text',
                labels_key='reals_text_ids',
                pad_token_id=autoencoder.encoder_text.tokenizer.pad_token_id,
                weight=loss_config['cross_entropy']['weights']['text'],
            )]
        if loss_config.get('latent_align', None):
            self.gen_loss_modules +=[
                LatentAlignmentLoss('shared_latents_audio_mean', 'shared_latents_text_mean', name='latent_align', weight=loss_config['latent_align']['weights']['text_image'])
            ]

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        if self.use_disc:
            self.disc_loss_modules = [
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ]

            self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None

        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )


        self.validation_step_outputs = []


    def configure_optimizers(self):
        gen_params = list(self.autoencoder.parameters())

        if self.use_disc:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())
            if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['autoencoder']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def forward(self, prompts):
        latents_text, encoder_info = self.autoencoder.encode(prompts, return_info=True)
        decoded_audio = self.autoencoder.decode(latents_text)
        return decoded_audio

    def validation_step(self, batch, batch_idx):
        reals_audio, metadata = batch

        prompts = [item['prompt'] for item in metadata]

        # Remove extra dimension added by WebDataset
        if reals_audio.ndim == 4 and reals_audio.shape[0] == 1:
            reals_audio = reals_audio[0]

        if len(reals_audio.shape) == 2:
            reals_audio = reals_audio.unsqueeze(1)

        loss_info = {}

        loss_info["reals_audio"] = reals_audio

        encoder_input = reals_audio

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        with torch.no_grad():
            latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, encoder_info = self.autoencoder.encode(prompts, audio=encoder_inputs, return_info=True, return_all=True)
            loss_info["latents_text_audio"] = latents_text_audio
            loss_info["latents_audio_audio"] = latents_audio_audio
            loss_info["latents_audio_text"] = latents_audio_text
            loss_info["latents_text_test"] = latents_text_test
            loss_info.update(encoder_info)

            decoded_audio_from_text, decoded_audio_from_audio, decoded_text_from_audio, decoded_text_from_text = self.autoencoder.decode(latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text)
            #Trim output to remove post-padding.
            decoded_audio_from_text, decoded_audio_from_audio, reals_audio = trim_to_shortest_triple(decoded_audio_from_text, decoded_audio_from_audio, reals_audio)

            # Run evaluation metrics.

        return val_loss_dict

    def on_validation_epoch_end(self):
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = value
                else:
                    sum_loss_dict[key] += value

        for key, value in sum_loss_dict.items():
            val_loss = value / len(self.validation_step_outputs)
            val_loss = self.all_gather(val_loss).mean().item()
            log_metric(self.logger, f"val/{key}", val_loss)

        self.validation_step_outputs.clear()  # free memory

    # ********************************************************
    def training_step(self, batch, batch_idx):
        reals_audio, metadata = batch

        prompts = [item['prompt'] for item in metadata]


        log_dict = {}
        # Remove extra dimension added by WebDataset
        if reals_audio.ndim == 4 and reals_audio.shape[0] == 1:
            reals_audio = reals_audio[0]

        if len(reals_audio.shape) == 2:
            reals_audio = reals_audio.unsqueeze(1)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True


        use_disc = (
            self.use_disc
            and self.global_step % 4 in [1,3]
            # Check warmup mode and if it is time to use discriminator.
            and (
                (self.warmup_mode == "full" and self.warmed_up)
                or self.warmup_mode == "adv")
        )
        if use_disc:
            for p in self.discriminator.discriminators.parameters():
                p.requires_grad =True
            self.discriminator.discriminators.train()
            for p in self.autoencoder.parameters():
                p.requires_grad =False
            self.autoencoder.eval()
        else:
            for p in self.discriminator.discriminators.parameters():
                p.requires_grad = False
            self.discriminator.discriminators.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad =True
            self.autoencoder.train()


        loss_info = {}

        loss_info["reals_audio"] = reals_audio

        encoder_input = reals_audio

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        # =======================================Step1. Encoder
        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, encoder_info = self.autoencoder.encode(prompts, audio=encoder_input, return_info=True, return_all=True)
        else:
            latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, encoder_info = self.autoencoder.encode(prompts, audio=encoder_input, return_info=True, return_all=True)

        loss_info.update(encoder_info)

        # =======================================Step2. Decoder
        decoded_audio_from_text, decoded_audio_from_audio, decoded_text_from_audio, decoded_text_from_text = self.autoencoder.decode(latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, decoder_text_freeze=self.decoder_text_freeze)

        #Trim output to remove post-padding
        decoded_audio_from_text, decoded_audio_from_audio, reals_audio = trim_to_shortest_triple(decoded_audio_from_text, decoded_audio_from_audio, reals_audio)

        # =======================================Step4. Calculate Loss      
        loss_info["decoded_audio_from_text"] = decoded_audio_from_text # Gen 3,6
        loss_info["decoded_audio_from_audio"] = decoded_audio_from_audio # Gen 3,6
        loss_info["decoded_text_logits_from_audio"] = decoded_text_from_audio # Gen 8
        loss_info["decoded_text_logits_from_text"] = decoded_text_from_text # Gen 8
        loss_info["reals_audio"] = reals_audio # Gen 3,6

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_audio_left_from_text"] = decoded_audio_from_text[:, 0:1, :] # Gen 4
            loss_info["decoded_audio_right_from_text"] = decoded_audio_from_text[:, 1:2, :] # Gen 5
            loss_info["decoded_audio_left_from_audio"] = decoded_audio_from_audio[:, 0:1, :] # Gen 4
            loss_info["decoded_audio_right_from_audio"] = decoded_audio_from_audio[:, 1:2, :] # Gen 5
            loss_info["reals_audio_left"] = reals_audio[:, 0:1, :] # Gen 4
            loss_info["reals_audio_right"] = reals_audio[:, 1:2, :] # Gen 5


        if self.use_disc:
            if self.warmed_up:
                loss_dis_from_text, loss_adv_from_text, feature_matching_distance_from_text = self.discriminator.loss(reals=reals_audio, fakes=decoded_audio_from_text.detach() if self.separate_disc and use_disc else decoded_audio_from_text)
                loss_dis_from_audio, loss_adv_from_audio, feature_matching_distance_from_audio = self.discriminator.loss(reals=reals_audio, fakes=decoded_audio_from_audio.detach() if self.separate_disc and use_disc else decoded_audio_from_audio)
            else:
                loss_adv_from_text = torch.tensor(0.).to(reals_audio)
                loss_adv_from_audio = torch.tensor(0.).to(reals_audio)
                feature_matching_distance_from_text = torch.tensor(0.).to(reals_audio)
                feature_matching_distance_from_audio = torch.tensor(0.).to(reals_audio)

                if self.warmup_mode == "adv":
                    loss_dis_from_text, _, _ = self.discriminator.loss(reals=reals_audio, fakes=decoded_audio_from_text.detach() if self.separate_disc and use_disc else decoded_audio_from_text)
                    loss_dis_from_audio, _, _ = self.discriminator.loss(reals=reals_audio, fakes=decoded_audio_from_audio.detach() if self.separate_disc and use_disc else decoded_audio_from_audio)
                else:
                    loss_dis_from_text = torch.tensor(0.0).to(reals_audio)
                    loss_dis_from_audio = torch.tensor(0.0).to(reals_audio)

            loss_info["loss_dis"] = (loss_dis_from_text + loss_dis_from_audio) *0.5
            loss_info["loss_adv"] = (loss_adv_from_text + loss_adv_from_audio) *0.5
            loss_info["feature_matching_distance"] = (feature_matching_distance_from_text + feature_matching_distance_from_audio)*0.5

        opt_gen = None
        opt_disc = None

        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            if self.use_disc:
                sched_gen, sched_disc = lr_schedulers
            else:
                sched_gen = lr_schedulers

        # Train the discriminator
        if use_disc:
            loss, losses = self.losses_disc(loss_info)

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm_disc > 0.0:
                if hasattr(self.trainer.precision_plugin, "scaler"):
                    self.trainer.precision_plugin.scaler.unscale_(opt_disc)  # <- here!
                pre_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm_disc)
                if self.clip_grad_print>1 and self.trainer.global_rank==0 and self.global_step % self.clip_grad_print==1:
                    print(f"||Discriminator||pre={float(pre_norm):.2f}  max_norm={self.clip_grad_norm_disc} clipped={float(pre_norm)>self.clip_grad_norm_disc}") 
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:
        
            # Text : Self ,Cross Generation
            # Audio : Self,Cross Generation
            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm_gen > 0.0:
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.clip_grad_norm_gen)
                if hasattr(self.trainer.precision_plugin, "scaler"):
                    self.trainer.precision_plugin.scaler.unscale_(opt_gen)  # <- here!
                pre_norm = torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.clip_grad_norm_gen)
                if self.clip_grad_print>1 and self.trainer.global_rank==0 and self.global_step % self.clip_grad_print==0:
                    print(f"||Generator||pre={float(pre_norm):.2f}  max_norm={self.clip_grad_norm_gen} clipped={float(pre_norm)>self.clip_grad_norm_gen}") 
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict['train/loss'] =  loss.detach().item()
            log_dict['train/latent_audio_text_std'] = latents_audio_text.std().detach().item()
            log_dict['train/latent_audio_audio_std'] = latents_audio_audio.std().detach().item()
            log_dict['train/latent_text_audio_std'] = latents_text_audio.std().detach().item()
            log_dict['train/latent_text_text_std'] = latents_text_text.std().detach().item()
            log_dict['train/data_std'] = data_std.detach().item()
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach().item()

        self.log_dict(log_dict, prog_bar=True, on_step=True)


        return loss
    # ********************************************************

    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class MultiModalVAEDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        sample_size=65536,
        sample_rate=44100,
        max_demos = 10,
        demo_dir=None,
        wandb_skip=False,
        demo_text=True,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(deepcopy(demo_dl))
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos
        self.demo_dir=demo_dir
        self.wandb_skip=wandb_skip
        self.demo_text=demo_text

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.demo_dir ==None:
            Path('./output_demo').mkdir(parents=True, exist_ok=True)
        else:
            Path(self.demo_dir).mkdir(parents=True, exist_ok=True)

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        module.eval()

        try:
            demo_reals_audio, metadata = next(self.demo_dl)

            prompts = [item['prompt'] for item in metadata]

            # Remove extra dimension added by WebDataset
            if demo_reals_audio.ndim == 4 and demo_reals_audio.shape[0] == 1:
                demo_reals_audio = demo_reals_audio[0]

            # Limit the number of demo samples
            if demo_reals_audio.shape[0] > self.max_demos:
                demo_reals_audio = demo_reals_audio[:self.max_demos,...]
                prompts = prompts[:self.max_demos]

            encoder_input = demo_reals_audio
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals_audio = demo_reals_audio.to(module.device)

            if not self.demo_text:
                with torch.no_grad():
                    if module.use_ema:
                        latents_text = module.autoencoder_ema.ema_model.encode(prompts)
                        fakes_audio = module.autoencoder_ema.ema_model.decode(latents_text)
                    else:
                        latents_text = module.autoencoder.encode(prompts)
                        fakes_audio = module.autoencoder.decode(latents_text)

                fakes_audio, demo_reals_audio = trim_to_shortest(fakes_audio.detach(), demo_reals_audio)

                #Interleave reals_audio and fakes_audio
                reals_audio_fakes = rearrange([demo_reals_audio, fakes_audio], 'i b d n -> (b i) d n')
                # Put the demos together
                reals_audio_fakes = rearrange(reals_audio_fakes, 'b d n -> d (b n)')
                
                try:
                    if self.demo_dir ==None:
                        filename = f'output_demo/recon_{trainer.global_step:08}.wav'
                    else:
                        filename = f'{self.demo_dir}/recon_{trainer.global_step:08}.wav'
                except:
                    filename = f'recon_{trainer.global_step:08}.wav'

                reals_audio_fakes = reals_audio_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reals_audio_fakes, self.sample_rate)

                #log_dict = {}
                if module.discriminator is not None:
                    window = torch.kaiser_window(512).to(fakes_audio.device)
                    fakes_audio_stft = torch.stft(fold_channels_into_batch(fakes_audio), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    fakes_audio_stft.requires_grad = True
                    fakes_audio_signal = unfold_channels_from_batch(torch.istft(fakes_audio_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), fakes_audio.shape[1])

                    real_stft = torch.stft(fold_channels_into_batch(demo_reals_audio), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    reals_audio_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals_audio.shape[1])
                    _, loss, _ = module.discriminator.loss(reals_audio_signal,fakes_audio_signal)

                    fakes_audio_stft.retain_grad()
                    loss.backward()
                    grads = unfold_channels_from_batch(fakes_audio_stft.grad.detach().abs(),fakes_audio.shape[1])

                    log_image_tensorboard(trainer.logger, grads, name="discriminator_sensitivity", step=trainer.global_step)

                    opts = module.optimizers()
                    opts[0].zero_grad()
                    opts[1].zero_grad()

            else:
                with torch.no_grad():
                    if module.use_ema:
                        latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, encoder_info = module.autoencoder_ema.ema_model.encode(prompts, audio=encoder_input, return_info=True, return_all=True)
                        decoded_audio_from_text, decoded_audio_from_audio, decoded_text_from_audio, decoded_text_from_text = module.autoencoder_ema.ema_model.decode(latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text)
                    else:
                        latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text, encoder_info = module.autoencoder.encode(prompts, audio=encoder_input, return_info=True, return_all=True)
                        decoded_audio_from_text, decoded_audio_from_audio, decoded_text_from_audio, decoded_text_from_text = module.autoencoder.decode(latents_text_audio, latents_audio_audio, latents_audio_text, latents_text_text)
                decoded_audio_from_text, decoded_audio_from_audio, demo_reals_audio = trim_to_shortest_triple(decoded_audio_from_text, decoded_audio_from_audio, demo_reals_audio)
                decoded_text_from_audio = module.autoencoder.decoder_text.generate(decoded_text_from_audio)
                decoded_text_from_text = module.autoencoder.decoder_text.generate(decoded_text_from_text)

                reals_audio_fakes_from_text = rearrange([demo_reals_audio, decoded_audio_from_text], 'i b d n -> (b i) d n')
                # Put the demos together
                reals_audio_fakes_from_text = rearrange(reals_audio_fakes_from_text, 'b d n -> d (b n)')
                try:
                    if self.demo_dir ==None:
                        filename = f'output_demo/recon_text_{trainer.global_step:08}.wav'
                    else:
                        filename = f'{self.demo_dir}/recon_text_{trainer.global_step:08}.wav'
                except:
                    filename = f'recon_text_{trainer.global_step:08}.wav'

                reals_audio_fakes_from_text = reals_audio_fakes_from_text.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reals_audio_fakes_from_text, self.sample_rate)

                reals_audio_fakes_from_audio = rearrange([demo_reals_audio, decoded_audio_from_audio], 'i b d n -> (b i) d n')
                reals_audio_fakes_from_audio = rearrange(reals_audio_fakes_from_audio, 'b d n -> d (b n)')
                try:
                    if self.demo_dir ==None:
                        filename = f'output_demo/recon_audio_{trainer.global_step:08}.wav'
                    else:
                        filename = f'{self.demo_dir}/recon_audio_{trainer.global_step:08}.wav'
                except:
                    filename = f'recon_audio_{trainer.global_step:08}.wav'

                reals_audio_fakes_from_audio = reals_audio_fakes_from_audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reals_audio_fakes_from_audio, self.sample_rate)
                print("[{:08}] Demo Text from Audio : {}".format(trainer.global_step, decoded_text_from_audio))
                print("[{:08}] Demo Text from Text : {}".format(trainer.global_step, decoded_text_from_text))

                #log_dict = {}
                if module.discriminator is not None:
                    window = torch.kaiser_window(512).to(decoded_audio_from_text.device)
                    decoded_audio_from_text_stft = torch.stft(fold_channels_into_batch(decoded_audio_from_text), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    decoded_audio_from_text_stft.requires_grad = True
                    decoded_audio_from_text_signal = unfold_channels_from_batch(torch.istft(decoded_audio_from_text_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), decoded_audio_from_text.shape[1])

                    real_stft = torch.stft(fold_channels_into_batch(demo_reals_audio), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    reals_audio_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals_audio.shape[1])
                    _, loss, _ = module.discriminator.loss(reals_audio_signal,decoded_audio_from_text_signal)

                    decoded_audio_from_text_stft.retain_grad()
                    loss.backward()
                    grads = unfold_channels_from_batch(decoded_audio_from_text_stft.grad.detach().abs(),decoded_audio_from_text.shape[1])
                    log_image_tensorboard(trainer.logger, grads, name="discriminator_sensitivity_from_text", step=trainer.global_step)

                    opts = module.optimizers()
                    opts[0].zero_grad()
                    opts[1].zero_grad()

                    window = torch.kaiser_window(512).to(decoded_audio_from_audio.device)
                    decoded_audio_from_audio_stft = torch.stft(fold_channels_into_batch(decoded_audio_from_audio), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    decoded_audio_from_audio_stft.requires_grad = True
                    decoded_audio_from_audio_signal = unfold_channels_from_batch(torch.istft(decoded_audio_from_audio_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), decoded_audio_from_audio.shape[1])

                    real_stft = torch.stft(fold_channels_into_batch(demo_reals_audio), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                    reals_audio_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals_audio.shape[1])
                    _, loss, _ = module.discriminator.loss(reals_audio_signal,decoded_audio_from_audio_signal)

                    decoded_audio_from_audio_stft.retain_grad()
                    loss.backward()
                    grads = unfold_channels_from_batch(decoded_audio_from_audio_stft.grad.detach().abs(),decoded_audio_from_audio.shape[1])
                    log_image_tensorboard(trainer.logger, grads, name="discriminator_sensitivity_from_audio", step=trainer.global_step)

                    opts = module.optimizers()
                    opts[0].zero_grad()
                    opts[1].zero_grad()


        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def create_loss_modules_from_bottleneck(bottleneck, loss_config, klkey="kl"):
    losses = []

    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key=klkey, weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    return losses
