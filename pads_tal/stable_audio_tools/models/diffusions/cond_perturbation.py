import numpy as np
import torch
import typing as tp

class ConditionPerturbation(object):
    _mode_debug_printed = False  
    
    def __init__(self,
            mixing_factor=1.0,
            max_dim=128,
            noise_scale=0.06, 
            tau1=0.85,
            tau2=1.0,
            use_annealing=False,
            rescale=False,
            scheduler_differential=1,
            schedule:tp.Literal["linear","step","cosine","polynomial"]="linear",
            mode:tp.Literal["original","cads","pads","reverse_pads"]="original",
            random_seed:int=None,
            pert_lmin:int=0):
        self.schedule = schedule
        self.use_annealing = use_annealing
        self.mixing_factor = mixing_factor
        self.noise_scale = noise_scale
        self.tau1 = tau1
        self.tau2 = tau2
        self.rescale = rescale
        self.max_dim = max_dim
        self.mode = mode
        self.scheduler_differential = scheduler_differential
        self.lmin = pert_lmin  
        
        # Independent random seed
        self.random_seed = random_seed if random_seed is not None else np.random.randint(0, 2**32 - 1)
        self._generators = {}
        print(f"[DEBUG] ConditionPerturbation initialized with random seed: {self.random_seed}, Lmin: {self.lmin}")
    
    def _get_generator(self, device):
        device_key = str(device)
        if device_key not in self._generators:
            self._generators[device_key] = torch.Generator(device=device)
            self._generators[device_key].manual_seed(self.random_seed)
        return self._generators[device_key]


    def linear_schedule(self, t):
        gamma = torch.where(t<=self.tau1, torch.ones_like(t), 
                        torch.where(t>=self.tau2, torch.zeros_like(t), 
                        (self.tau2-t)/(self.tau2-self.tau1)))
        return gamma
    
    def step_schedule(self, t):
        gamma = torch.where(t <= self.tau1, torch.ones_like(t), torch.zeros_like(t))
        return gamma
    
    def cosine_schedule(self, t, d=1):
        normalized_t = (t - self.tau1) / (self.tau2 - self.tau1)
        gamma = torch.cos(normalized_t * torch.pi / 2) ** d
        gamma = torch.where(t <= self.tau1, torch.ones_like(t),
                   torch.where(t >= self.tau2, torch.zeros_like(t), gamma))
        return gamma
    
    def polynomial_schedule(self, t, d=1):
        gamma = torch.where(t<=self.tau1, torch.ones_like(t),
                   torch.where(t>=self.tau2, torch.zeros_like(t), ((1-t)/(1-self.tau1))**d))
        return gamma
    
    def add_noise(self, y, attn_mask, t):
        """ Add noise to the condition
        
        Arguments:
        y: Input conditioning : [Batch, concated length=80, cond_dim=768]
        t : Time step : [Batch], 
        attn_mask: conditioning mask : [batch, concated length=80] 0 or 1
        audio_input: audio input for attention calculation : [batch, seq_len, embed_dim]
        attention_module: attention module for calculating attention scores
        """
        if not ConditionPerturbation._mode_debug_printed:
            print(f"[DEBUG] ConditionPerturbation mode: {self.mode}, schedule: {self.schedule}")
            ConditionPerturbation._mode_debug_printed = True

        if self.schedule == "linear":
            gamma = self.linear_schedule(t)
        elif self.schedule == "step":
            gamma = self.step_schedule(t)
        elif self.schedule == "cosine":
            gamma = self.cosine_schedule(t, self.scheduler_differential)
        elif self.schedule == "polynomial":
            gamma = self.polynomial_schedule(t, self.scheduler_differential)
        else:
            print("[ConditionPerturbation][Error] Invalid schedule: {}".format(self.schedule))
            exit()

        gamma_sqrt = torch.sqrt(gamma).view(-1, 1, 1)
        one_minus_gamma_sqrt = torch.sqrt(1.0 - gamma).view(-1, 1, 1)

        if y.shape[1] < self.max_dim:
            print("[ConditionPerturbation][Warning] Cross Attn dim {} is less than max_dim {}".format(y.shape[1], self.max_dim))
            return y
        
        y_kept = y[:, self.max_dim:, :]  # stable audio open additional conditioning: Start time, End time, BPM
        y = y[:, :self.max_dim, :]  # Text [2,128,768]

        if self.rescale:
            y_mean, y_std = torch.mean(y), torch.std(y)
        
        if self.mode == "pads":
            if y.shape[0] != attn_mask.shape[0]:
                if attn_mask.shape[0] == 1:
                    attn_mask = attn_mask.expand(y.shape[0], -1)
                else:
                    attn_mask = attn_mask[:1].expand(y.shape[0], -1)

            attn_mask = attn_mask[:, :self.max_dim]
            mask = attn_mask.unsqueeze(-1).float()  # [B, S, 1]
            
            if self.lmin > 0:
                modified_mask = mask.clone()
                modified_mask[:, -self.lmin:, :] = 0.0
            else:
                modified_mask = mask
            
            noise = torch.randn(y.shape, device=y.device, generator=self._get_generator(y.device)) * self.noise_scale
            if self.use_annealing:
                y = y * modified_mask + (1.0 - modified_mask) * (gamma_sqrt * y + one_minus_gamma_sqrt * noise)
            else:
                y = y * modified_mask + noise * (1.0 - modified_mask)
                
        elif self.mode == "reverse_pads":
            if y.shape[0] != attn_mask.shape[0]:
                if attn_mask.shape[0] == 1:
                    attn_mask = attn_mask.expand(y.shape[0], -1)
                else:
                    attn_mask = attn_mask[:1].expand(y.shape[0], -1)

            attn_mask = attn_mask[:, :self.max_dim]
            mask = attn_mask.unsqueeze(-1).float()  # [B, S, 1]
            
            if self.lmin > 0:
                modified_mask = mask.clone()
                modified_mask[:, -self.lmin:, :] = 1.0
            else:
                modified_mask = mask

            noise = torch.randn(y.shape, device=y.device, generator=self._get_generator(y.device)) * self.noise_scale
            if self.use_annealing:
                y = y * (1.0 - modified_mask) + modified_mask * (gamma_sqrt * y + one_minus_gamma_sqrt * noise)
            else:
                y = y * (1.0 - modified_mask) + noise * modified_mask
                
        elif self.mode == "cads":
            y = gamma_sqrt * y + self.noise_scale * one_minus_gamma_sqrt * torch.randn(y.shape, device=y.device, generator=self._get_generator(y.device))
        
        else:
            print("[ConditionPerturbation][Error] Invalid mode: {}".format(self.mode))
            exit()
            
        if self.rescale:
            y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
            if not torch.isnan(y_scaled).any():
                if self.mixing_factor != 1.0:
                    y = self.mixing_factor * y_scaled + (1 - self.mixing_factor) * y
                else:
                    y = y_scaled
            else:
                print("[ConditionPerturbation][Warning] NaN encountered in rescaling")
        
        return torch.cat([y, y_kept], dim=1)

    def __call__(self, text_cond, attn_mask, sampling_step):
        results = self.add_noise(text_cond, attn_mask, sampling_step)
        return results

