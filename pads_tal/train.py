from prefigure.prefigure import get_all_args, push_wandb_config
import argparse
import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import random
import datetime

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from tools.model_utils import get_model_from_ckpt, load_model_from_ckpt

os.environ['WANDB_DISABLED'] = 'true'
class ExceptionCallback3(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback4(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config
        model_config_type = model_config.get('model_type', "")
        model_config_training = model_config.get('training', None)
        self.model_config_ema = model_config_training.get('use_ema', False)
        self.save_type=None
        if model_config_type=='autoencoder':
            self.save_type='autoencoder'

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.save_type=="autoencoder":
            state_dict_autoencoder={}

            for k, v in checkpoint['state_dict'].items():
                # Save Autoencoder Only
                if k.startswith('autoencoder.'):
                    state_dict_autoencoder.update({k: v})
                    continue;
                # Save Autoencoder Only - EMA
                if self.model_config_ema and k.startswith('autoencoder_ema.ema_model.'):
                    state_dict_autoencoder.update({k:v })
                    continue;
                state_dict_autoencoder.update({k: v}) # Else
            checkpoint['state_dict']=state_dict_autoencoder
        checkpoint["model_config"] = self.model_config

def main():
    parser = argparse.ArgumentParser(description='Run Trainer')
    parser.add_argument('--name', type=str, help='', default="stable_audio_tools", required=False)
    parser.add_argument('--batch-size', type=int, help='', default=8, required=False)
    parser.add_argument('--num-gpus', type=int, help='', default=1, required=False)
    parser.add_argument('--num-nodes', type=int, help='', default=1, required=False)
    parser.add_argument('--strategy', type=str, help='', default="", required=False)
    parser.add_argument('--precision', type=str, help='', default="16-mixed", required=False)
    parser.add_argument('--epochs', type=int, help='', default=-1, required=False) # Infinite
    parser.add_argument('--steps', type=int, help='', default=-1, required=False)
    parser.add_argument('--num-workers', type=int, help='', default=8, required=False)
    parser.add_argument('--seed', type=int, help='', default=42, required=False)
    parser.add_argument('--accum-batches', type=int, help='', default=1, required=False)
    parser.add_argument('--checkpoint-every', type=int, help='', default=20000, required=False)
    parser.add_argument('--pretrained-ckpt-path', type=str, default="",help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, default="", help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--pretransform-load-ema', action='store_true', default=False, help='', required=False)
    parser.add_argument('--model-config', type=str, default="", help='Path to model config', required=False)
    parser.add_argument('--dataset-config', type=str, default="", help='Name of pretrained model', required=True)
    parser.add_argument('--save-dir', type=str, default="./output_ckpt", help='', required=False)
    parser.add_argument('--gradient-clip-val', type=float, default=0.0,help='', required=False)
    parser.add_argument('--remove-pretransform-weight-norm', type=str, default="./output_ckpt", help='', required=False)
    parser.add_argument('--wandb-skip', action='store_true', default=True, help='', required=False)
    parser.add_argument('--resume-from', type=str, choices=['none', 'base'], default='none', help="Where you want to resume from", required=False)
    parser.add_argument('--local_rank', type=int, help='', required=False)
    args = parser.parse_args()

    wandb_skip=args.wandb_skip
    if not wandb_skip:
        args = get_all_args()

    seed = args.seed
    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))
    random.seed(seed)
    torch.manual_seed(seed)


    # 1. model ==================================================
    ckpt_return=False
   
    if args.resume_from=="base": # Resume Training
        if not args.pretrained_ckpt_path or not args.pretrained_ckpt_path.endswith('.ckpt'): # Finetuning Training
            raise Exception("[Error] if you want to resume from base model, give --pretrained-ckpt-path : {}".format(args.pretrained_ckpt_path))
        
        MODEL, model_config, ckpt_return = get_model_from_ckpt(
            args.pretrained_ckpt_path, 
            config_path=args.model_config, 
            print_remain=True, 
            ema_enabled=False,) 
    elif args.pretrained_ckpt_path: 
        MODEL, model_config, ckpt_return = get_model_from_ckpt(
            args.pretrained_ckpt_path, 
            config_path=args.model_config, 
            print_remain=True, 
            ema_enabled=False,) 
    else: # Scratch Training
        if args.model_config=="":
            raise Exception("[Error] Model Config Required for scratch training")
        #Get JSON config from args.model_config
        with open(args.model_config) as f:
            model_config = json.load(f)

        MODEL = create_model_from_config(model_config)
    
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(MODEL.pretransform)

    if args.pretransform_ckpt_path:
        MODEL, _ = load_model_from_ckpt(
                MODEL, args.pretransform_ckpt_path, 
                print_remain=True, 
                ema_inference=args.pretransform_load_ema, 
                only_mode="autoencoder")
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(MODEL.pretransform)



    TRAIN_WRAPPER = create_training_wrapper_from_config(model_config, MODEL)


    # 2. data ==================================================
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    DATALOADER = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )
    print("\n\n[Info] Data Config Check : {}\n{}\n\n".format(dataset_config, DATALOADER))


    # 3. Logger & Callback ===========================================

    callback3_exc = ExceptionCallback3()
    
    datetag = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    datetag_path = "{}/{}".format(args.name, datetag)

    if wandb_skip :
        wandb_logger=TensorBoardLogger("./output_ckpt/lightning_logs", name=f"{datetag_path}"),
    else : 
        wandb_logger = pl.loggers.WandbLogger(project=args.name)
        wandb_logger.watch(TRAIN_WRAPPER)

    if args.save_dir and wandb_skip:
        checkpoint_dir = os.path.join(args.save_dir, datetag_path) 
    elif args.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None

    callback1_ckpt = pl.callbacks.ModelCheckpoint(
                    every_n_train_steps=args.checkpoint_every, 
                    dirpath=checkpoint_dir, 
                    save_top_k=-1)
    callback4_save_model_config = ModelConfigEmbedderCallback4(model_config)


    demo_dir = os.path.join('./output_demo', datetag_path) 
    callback2_demo = create_demo_callback_from_config(model_config, demo_dir=demo_dir, demo_dl=DATALOADER, wandb_skip=wandb_skip)

    #Combine args and config dicts
    if not wandb_skip : 
        args_dict = vars(args)
        args_dict.update({"model_config": model_config})
        args_dict.update({"dataset_config": dataset_config})
        push_wandb_config(wandb_logger, args_dict)

    # 4. Training Setting ===========================================
    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
            accelerator="gpu"
        elif args.strategy =="horovod":
            strategy=None
            accelerator="horovod"
        else:
            strategy = args.strategy
            accelerator="gpu"
    else:

        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 
        accelerator="gpu"

    if args.steps!=-1:
        args.epochs=None

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator=accelerator,
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        logger=wandb_logger,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[callback1_ckpt, callback2_demo, callback3_exc, callback4_save_model_config],
        log_every_n_steps=1,
        max_epochs=args.epochs,
        max_steps=args.steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0
    )


    resume_point=None
    if args.resume_from=="base":
        resume_point=args.pretrained_ckpt_path
        print("\n[Info] Resume from {}\n".format(resume_point))
    else:
        print("\n[Info] No Resume Mode\n")

    trainer.fit(
        TRAIN_WRAPPER, 
        DATALOADER, 
        ckpt_path=resume_point
    )


if __name__ == '__main__':

    main()
