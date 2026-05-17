import json

def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .model_autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    if model_type == 'autoencoder_mvae':
        from .model_mvae import create_mvae_from_config
        return create_mvae_from_config(model_config)
    elif model_type == 'diffusion_cond':
        from .model_diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_model_from_config_path(model_config_path):
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    return create_model_from_config(model_config)


