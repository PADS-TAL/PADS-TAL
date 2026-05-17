import torch 
import gc
import os
import json
from torch.nn.functional import mse_loss
import numpy as np
from stable_audio_tools import create_model_from_config
import typing as tp


def copy_state_dict_autoencoder(model, state_dict, first_remove=False, print_remain=False, ema_inference=False):
    state_dict_autoencoder={} 
    for key, value in state_dict.items():
        key_list = key.split('.')
        if ema_inference :
            if not (key_list[0]=="autoencoder_ema" and key_list[1]=="ema_model"):
                continue;
            if first_remove:
                model_key = '.'.join(key_list[2:])
            else:
                model_key='.'.join(key_list[1:])
        else:
            if key_list[0]=="autoencoder_ema":
                continue;
            if first_remove:
                model_key = '.'.join(key_list[1:])
            else:
                model_key=key
        if model_key.startswith('encoder.') or model_key.startswith('decoder.') or model_key.startswith('encoder_text.') or model_key.startswith('private_audio.') or model_key.startswith('decoder_audio.'):
            state_dict_autoencoder[model_key] = value

    model.pretransform.model.load_state_dict(state_dict_autoencoder, strict=False)

    autoencoder_state_dict_check = model.pretransform.model.state_dict()
    if print_remain:
        count = 0
        for model_key in autoencoder_state_dict_check:
            if model_key not in state_dict_autoencoder:
                count +=1
                print("[Autoencoder][Warning] Model Not Loaded : pretransform.model.{}".format(model_key))
        if len(state_dict_autoencoder)==0 :
            raise Exception("[Autoencoder][Error] Autoencoder Nothing to load from state_dict to model\n")
        elif count==0:
            print("[Autoencoder][Info] Autoencoder Successfully Loaded\n")
        else:
            print("[Autoencoder][Info] Autoencoder Done with something unloaded\n")


def copy_state_dict(model, state_dict, first_remove=False, print_remain=False, specific=[], ema_inference=False):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
        specific :  "model.model.to_timestep_embed"
                    "model.model.to_cond_embed"
                    "model.model.to_global_embed"
                    "model.model.transformer"
                    "model.model.preprocess"
                    "model.model.postprocess"
                    "conditioners.conditioners.seconds_start"
                    "conditioners.conditioners.seconds_total"
                    "pretransform.model.encoder"
                    "pretransform.model.decoder"
    """
    if print_remain:
        remove_test=[]
    model_state_dict = model.state_dict()
    for key in state_dict:
        key_list = key.split('.')
        if ema_inference and not key_list[1] in ["pretransform", 'conditioner']: # TODO : conditioner check
            if not (key_list[0]=="diffusion_ema") :
                continue;
            if first_remove:
                model_key = '.'.join([key_list[1].replace('ema_model','model')]+key_list[2:])
            else:
                model_key='.'.join([key_list[0].replace('ema_model','model')]+key_list[1:])
        else:
            if key_list[0]=="diffusion_ema" or key_list[0]=="autoencoder_ema":
                continue;
            if first_remove:
                model_key = '.'.join(key_list[1:])
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
                    model_state_dict[model_key] = state_dict[key]
                    if print_remain:
                        remove_test.append(model_key) #temp
                else :
                    key_compare2 = '.'.join(model_key.split('.')[:2])
                    key_compare3 = '.'.join(model_key.split('.')[:3])
                    if key_compare3 in specific or key_compare2 in specific:
                        model_state_dict[model_key] = state_dict[key]
                        if print_remain:
                            remove_test.append(model_key) #temp
            else:
                print("[Warning] StateDict weight Not match with Model(Size) : {}!={}".format(state_dict[key].shape, model_state_dict[key].shape)) # Nothing
        else:
            print("[Warning] StateDict weight Not match with Model (Key) : {}".format(key))
    if print_remain:
        count = 0
        for key in model_state_dict:
            if specific==[]:
                key_compare2 = '.'.join(key.split('.')[:2])
                key_compare3 = '.'.join(key.split('.')[:3])
                if key not in remove_test:
                    count +=1
                    print("[Warning] Model Not Loaded : {}".format(key))
            else:
                key_compare2 = '.'.join(key.split('.')[:2])
                key_compare3 = '.'.join(key.split('.')[:3])
                if key not in remove_test and (key_compare3 in specific or key_compare2 in specific) :
                    count +=1
                    print("[Warning] Model Not Loaded : {}".format(key))
        if len(remove_test)==0:
            print("[Warning] Nothing to load from state_dict to model\n")
        elif count==0:
            print("[Info] Successfully Loaded\n")
        else:
            print("[Info] Done with something unloaded\n")

    model.load_state_dict(model_state_dict, strict=False)


def get_model_from_ckpt(ckpt_path, config_path="", config_json=None, print_remain=False, specific=[], ema_enabled=False, save_name=''):

    ckpt_return=None
    first_remove=None
    if ckpt_path.endswith(".safetensors"):
        if config_path !="" :
            with open(config_path) as f:
                model_config = json.load(f)
        elif config_json!=None:
            model_config = config_json
        else:
            raise Exception("[Error] config_path must be required : {}".format(config_path))
        model = create_model_from_config(model_config)
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        
        ckpt_return=False
        first_remove=False
        ema_enabled=False
    elif ckpt_path.endswith(".ckpt"):
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if config_path!="" and config_path!=None :
            print("\n[Warning] config_path is not required, but i will load : {}\n".format(config_path))
            with open(config_path) as f:
                model_config = json.load(f)
        elif config_json!=None:
            model_config = config_json
        else : 
            model_config = loaded_ckpt['model_config']
        model = create_model_from_config(model_config)
        state_dict = loaded_ckpt["state_dict"]

        first_remove=True
        ckpt_return=True
    else:
        _, ext = os.path.splitext(ckpt_path)
        raise Exception("[Error] This format is not supported : {}".format(ext))

    model_config_type = model_config.get('model_type', "")
    model_config_training = model_config.get('training', None)
    model_config_ema = model_config_training.get('use_ema', False)
    if model_config_ema and ema_enabled:
        ema_inference=True # TODO
    else:
        ema_inference=False


    if model_config_type =="diffusion_cond":
        diffusion_config = model_config['model']['diffusion']['config']
        if 'attn_kwargs' in diffusion_config and 'save_kwargs' in diffusion_config['attn_kwargs']:
            if save_name=='':
                model_config['model']['diffusion']['config']['attn_kwargs']['save_kwargs'] = {}
            elif 'save_name' not in diffusion_config['attn_kwargs']['save_kwargs']:
                model_config['model']['diffusion']['config']['attn_kwargs']['save_kwargs']['save_name'] = save_name

    copy_state_dict(
            model, 
            state_dict, 
            first_remove=first_remove, 
            print_remain=print_remain, 
            specific=specific, 
            ema_inference=ema_inference)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return model, model_config, ckpt_return
#=================================================================V2

def load_model_from_ckpt(model, ckpt_path, print_remain=False, specific=[], ema_inference=False, 
                        only_mode:tp.Literal["None", "autoencoder"] ="None"):        

    ckpt_return=None
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        ckpt_return=False
        first_remove=False
    elif ckpt_path.endswith(".ckpt"):
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = loaded_ckpt["state_dict"]
        first_remove=True
        ckpt_return=True
    else:
        _, ext = os.path.splitext(ckpt_path)
        raise Exception("[Error] This format is not supported : {}".format(ext))

    if only_mode=="autoencoder":
        if ckpt_path.endswith(".safetensors"):
            state_dict_autoencoder={} 
            for model_key,value in state_dict.items():
                if model_key.startswith('pretransform.model.'):
                    state_dict_autoencoder[model_key.replace("pretransform.model.", "")] = value
            copy_state_dict_autoencoder(model, 
                                        state_dict_autoencoder, 
                                        first_remove=first_remove,
                                        print_remain=print_remain)
        elif ckpt_path.endswith(".ckpt"):
            model_config = loaded_ckpt['model_config']
            model_config_type = model_config.get('model_type', "")
            if model_config_type not in ["autoencoder", "autoencoder_mmvae" ]:
                raise Exception("[Error] Why Autoencoder Trained with weird Network : {}".format(model_config_type))
            model_config_training = model_config.get('training', None)
            model_config_ema = model_config_training.get('use_ema', False)
            copy_state_dict_autoencoder(model, 
                                        state_dict, 
                                        first_remove=first_remove,
                                        print_remain=print_remain,
                                        ema_inference=ema_inference & model_config_ema)
    else:
        copy_state_dict(
                model, 
                state_dict, 
                first_remove=first_remove, 
                print_remain=print_remain, 
                specific=specific,
                ema_inference=ema_inference)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
        
    return model, ckpt_return

