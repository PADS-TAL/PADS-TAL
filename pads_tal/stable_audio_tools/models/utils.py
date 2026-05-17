import torch
from safetensors.torch import load_file

from torch.nn.utils import remove_weight_norm

def copy_state_dict(model, state_dict, first_remove=0, print_remain=False, specific=[], ema_inference=False, controlnet_copy_load=False, print_name=""):
    if print_remain:
        remove_test=[]
    model_state_dict = model.state_dict()
    for key in state_dict:
        key_list = key.split('.')
        if ema_inference and not key_list[1] in ["pretransform", 'conditioner']: # TODO : conditioner check
            if not (key_list[0]=="diffusion_ema") :
                continue;
            model_key='.'.join([key_list[first_remove].replace('ema_model','model')]+key_list[first_remove+1:])
        else:
            if key_list[0]=="diffusion_ema" or key_list[0]=="autoencoder_ema":
                continue;
            if first_remove :
                model_key = '.'.join(key_list[first_remove:])
            else:
                model_key=key
        if model_key in model_state_dict :
            if state_dict[key].shape == model_state_dict[model_key].shape:
                if isinstance(state_dict[key], torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    state_dict[key] = state_dict[key].data
                if specific==[]:
                    key_compare2 = '.'.join(model_key.split('.')[:2])
                    key_compare3 = '.'.join(model_key.split('.')[:3])
                    if not (controlnet_copy_load and (key_compare2=="model.controlnet"or key_compare3=="conditioner.conditioners.melody")):
                        model_state_dict[model_key] = state_dict[key]
                        if print_remain:
                            remove_test.append(model_key) #temp
                else :
                    key_compare2 = '.'.join(model_key.split('.')[:2])
                    key_compare3 = '.'.join(model_key.split('.')[:3])
                    if key_compare3 in specific or key_compare2 in specific:
                        if not (controlnet_copy_load and (key_compare2=="model.controlnet"or key_compare3=="conditioner.conditioners.melody")):
                            model_state_dict[model_key] = state_dict[key]
                            if print_remain:
                                remove_test.append(model_key) #temp
            else:
                print("[Warning] StateDict weight Not match with Model(Size) : {}!={}".format(state_dict[key].shape, model_state_dict[key].shape)) # Nothing
        else:
            pass
    if print_remain:
        count = 0
        for key in model_state_dict:
            if specific==[]:
                key_compare2 = '.'.join(key.split('.')[:2])
                key_compare3 = '.'.join(key.split('.')[:3])
                if controlnet_copy_load and (key_compare2=="model.controlnet"or key_compare3=="conditioner.conditioners.melody"):
                    pass;
                elif key not in remove_test:
                    count +=1
                    print("[Warning] Model Not Loaded : {}".format(key))
            else:
                key_compare2 = '.'.join(key.split('.')[:2])
                key_compare3 = '.'.join(key.split('.')[:3])
                if controlnet_copy_load and (key_compare2=="model.controlnet"or key_compare3=="conditioner.conditioners.melody"):
                    pass;
                elif key not in remove_test and (key_compare3 in specific or key_compare2 in specific) :
                    count +=1
                    print("[Warning] Model Not Loaded : {}".format(key))
        if len(remove_test)==0:
            print("[Warning] Nothing to load from state_dict to model\n")
        elif count==0:
            print("[Info] Successfully Loaded : {}\n".format(print_name))
        else:
            print("[Info] Done with something unloaded : {}\n".format(print_name))

    model.load_state_dict(model_state_dict, strict=False)

def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    
    return state_dict

def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model

# Get torch.compile flag from environment variable ENABLE_TORCH_COMPILE

import os
enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):
    
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function

# Sampling functions copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/utils.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def next_power_of_two(n):
    return 2 ** (n - 1).bit_length()

def next_multiple_of_64(n):
    return ((n + 63) // 64) * 64
