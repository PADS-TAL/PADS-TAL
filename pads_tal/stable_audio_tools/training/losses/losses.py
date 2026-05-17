import typing as tp
import torch

from torch.nn import functional as F
from torch import nn

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, decay = 1.0):
        super().__init__()

        self.name = name
        self.decay = float(decay)
        weight = torch.tensor(float(weight))
        self.register_buffer('weight', weight)

    def decay_weight(self):
        if self.decay != 1.0:
            self.weight *= self.decay

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError

class L1LossDouble(LossModule):
    def __init__(self, key_a: str, key_b: str, key_b2:str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b
        self.key_b2 = key_b2

        self.mask_key = mask_key

    def forward(self, info):
        l1_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')
        l1_loss2 = F.l1_loss(info[self.key_a], info[self.key_b2], reduction='none')

        if self.mask_key is not None and self.mask_key in info:
            l1_loss = l1_loss[info[self.mask_key]]
            l1_loss2 = l1_loss2[info[self.mask_key]]

        l1_loss = l1_loss.mean()
        l1_loss2 = l1_loss2.mean()
        self.decay_weight()
        return self.weight * (l1_loss + l1_loss2) * 0.5

class AuralossLossDouble(LossModule):
    def __init__(self, loss_module, input_key: str, input_key2:str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(name, weight, decay)

        self.loss_module = loss_module
        self.input_key = input_key
        self.input_key2 = input_key2
        self.target_key = target_key

    def forward(self, info):
        loss = self.loss_module(info[self.target_key], info[self.input_key]) # Enforce wrong order of input and target until we find issue in Auraloss
        loss2 = self.loss_module(info[self.target_key], info[self.input_key2]) # Enforce wrong order of input and target until we find issue in Auraloss
        self.decay_weight()
        return self.weight * (loss + loss2) * 0.5

class LatentAlignmentLoss(LossModule):
    def __init__(self, key_a: str, key_b: str, name='latent_align', weight=1.0, mode='l2', decay=1.0):
        super().__init__(name=name, weight=weight, decay=decay)
        self.key_a = key_a
        self.key_b = key_b
        self.mode = mode

    def forward(self, info):
        a = info[self.key_a]  # e.g., z_audio
        b = info[self.key_b]  # e.g., z_text

        if self.mode == 'l2':
            loss = F.mse_loss(a, b)
        elif self.mode == 'cosine':
            loss = 1 - F.cosine_similarity(a, b, dim=-1).mean()
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')
        
        self.decay_weight()
        return self.weight * loss

class CrossEntropyTokenLossDouble(LossModule):
    def __init__(self, logits_key: str, logits_key2:str, labels_key: str, pad_token_id: int, weight: float = 1.0, name: str = 'cross_entropy', decay: float = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)
        self.logits_key = logits_key
        self.logits_key2 = logits_key2
        self.labels_key = labels_key
        self.pad_token_id = pad_token_id
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, info):
        logits = info[self.logits_key]   # [B, L, vocab]
        logits2 = info[self.logits_key2]   # [B, L, vocab]
        labels = info[self.labels_key]   # [B, L]

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss2 = self.loss_fct(logits2.view(-1, logits2.size(-1)), labels.view(-1))
        self.decay_weight()
        return self.weight * (loss+loss2)*0.5
class CrossEntropyTokenLoss(LossModule):
    def __init__(self, logits_key: str, labels_key: str, pad_token_id: int, weight: float = 1.0, name: str = 'cross_entropy', decay: float = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)
        self.logits_key = logits_key
        self.labels_key = labels_key
        self.pad_token_id = pad_token_id
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, info):
        logits = info[self.logits_key]   # [B, L, vocab]
        labels = info[self.labels_key]   # [B, L]
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.decay_weight()
        return self.weight * loss


class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0, decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key = key

    def forward(self, info):
        return self.weight * info[self.key]

class TargetValueLoss(LossModule):
    def __init__(self, key: str, target: float, name: str, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
        self.target = target
    
    def forward(self, info):
        return self.weight * (info[self.key] - self.target).abs()

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss', decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        l1_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info:
            l1_loss = l1_loss[info[self.mask_key]]

        l1_loss = l1_loss.mean()
        self.decay_weight()
        return self.weight * l1_loss

class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss',decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()
        self.decay_weight()
        return self.weight * mse_loss
class LossWithTarget(LossModule):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(name, weight, decay)

        self.loss_module = loss_module

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.loss_module(info[self.input_key], info[self.target_key])
        self.decay_weight()
        return self.weight * loss
class AuralossLoss(LossWithTarget):
    def __init__(self, loss_module, input_key: str, target_key: str, name: str, weight: float = 1, decay = 1.0):
        super().__init__(loss_module, input_key=input_key, target_key=target_key, name=name, weight=weight, decay=decay)
    def forward(self, info):
        loss = self.loss_module(info[self.target_key], info[self.input_key]) # Enforce wrong order of input and target until we find issue in Auraloss
        self.decay_weight()
        return self.weight * loss

class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info, except_loss=[]):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            if loss_module.name in except_loss:
                continue;
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses