import torch
import torchaudio
import datetime
import json
import os
#import re
#from unicodedata import normalize
import argparse
from pathlib import Path
import numpy as np
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from tools.model_utils import get_model_from_ckpt, load_model_from_ckpt
from tools.audio_utils import convert_wav_to_mp3 
from tools.eval_tools import _load_model as load_clap_model, calculate_in_batch_similarity, save_in_batch_similarity_results
import pandas as pd
from stable_audio_tools.data.utils import Stereo,PadCrop_Normalized_T
import pyloudnorm as pyln

from pedalboard.io import AudioFile
from torchaudio import transforms as T

import csv
import ast
from glob import glob
device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser(description='Run Inference')
# Model Path (Required)
parser.add_argument('--ckpt-path', type=str, help='Model Path to model checkpoint', required=True)
parser.add_argument('--model-config', type=str, default="", help='Path to model config', required=False)
parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)

# Data Path (Required
parser.add_argument('--dataset', choices=['songdescriber', 'melbench'], required=True)
parser.add_argument('--text-type', choices=['caption', 'tag'], default='tag', help='select data type')

# Inference
parser.add_argument('--sampler-type', type=str, default='k-dpm-2', choices=["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde"], required=False)
parser.add_argument('--batch-size', type=int, default=1, help='Give batch size', required=False)
parser.add_argument('--cfg-scale', type=int, default=7, help='cfg scale', required=False)
parser.add_argument('--steps', type=int, default=100, help='inference steps', required=False)
parser.add_argument('--seed', type=int, default=-1, help='seed', required=False)
parser.add_argument('--init-noise-level', type=float, default=0.1, help='init-noise-level', required=False)
parser.add_argument('--sigma-min', type=float, default=0.03, required=False)
parser.add_argument('--sigma-max', type=float, default=500.0, required=False)

# Save
parser.add_argument('--save-wav', action='store_true', help='Whether to save in wav format', required=False)
parser.add_argument('--save-name', type=str, default='noname', required=True)
parser.add_argument('--mmvae-cond', action='store_true', help='Whether to mmvae conditioning', required=False)
parser.add_argument('--mmvae-timestep-loc', type=int, default=1, help='Whether to mmvae conditioning', required=False)
parser.add_argument('--mmvae-init-multi', action='store_true', default=False, help='Whether to mmvae conditioning', required=False)
parser.add_argument('--mmvae-mix-alpha', type=float, default=0.6, help='Whether to mmvae conditioning', required=False)
parser.add_argument('--recursive', action='store_true', help='Whether to single', default=False, required=False)
parser.add_argument('--dataset-genre', type=str, default='', choices=['elec', 'jazz', 'pop','newage',
            'blues', 'classic', 'country', 'latin', 'metal','rock', 'easy', 'folk', 'rnb', 'hiphop', 'world'], required=False)
parser.add_argument('--in-batch-sim', action='store_true', help='Calculate in-batch similarity score', required=False)
parser.add_argument('--force-save', action='store_true', help='force save all files', required=False)
parser.add_argument('--single', action='store_true', help='Whether to single', required=False)

args = parser.parse_args()


if __name__ == '__main__':
    # ========================================================= Model 
    if  args.dataset == 'songdescriber': 
        csv_file_path = 'input_inf/eval/baseline_songdesc_tags.csv'  # TODO: add 'aspect_list' column
    elif args.dataset =='melbench':
        if args.text_type == 'tag':
            if args.dataset_genre:
                if args.dataset_genre == 'elec':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_electronic.csv'
                elif args.dataset_genre == 'jazz':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_jazz.csv'
                elif args.dataset_genre == 'pop':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_pop.csv'
                elif args.dataset_genre == 'newage':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_new_age.csv'
                elif args.dataset_genre == 'blues':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_blues.csv'
                elif args.dataset_genre == 'classic':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_classical.csv'
                elif args.dataset_genre == 'country':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_country.csv'
                elif args.dataset_genre == 'latin':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_latin.csv'
                elif args.dataset_genre == 'metal':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_metal.csv'
                elif args.dataset_genre == 'rock':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_rock.csv'
                elif args.dataset_genre == 'easy':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_easy_listening.csv'
                elif args.dataset_genre == 'folk':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_folk_acoustic.csv'
                elif args.dataset_genre == 'rnb':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_rnb.csv'
                elif args.dataset_genre == 'hiphop':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_hip_hop.csv'
                elif args.dataset_genre == 'world':
                    csv_file_path = 'input_inf/eval/baseline_melbench_genres/melbench_world_traditional.csv'
                else:
                    raise Exception("[Error] MelBench dataset for genre {} is not supported".format(args.dataset_genre))
            else:
                csv_file_path = 'input_inf/eval/baseline_melbench.csv'
        else:
            raise Exception("[Error] MelBench dataset for sentences not supported")


    if not args.ckpt_path:
        raise Exception("[Error] --ckpt-path must be given")
    
    elif args.ckpt_path.endswith(".ckpt") or args.ckpt_path.endswith(".safetensors"):
        MODEL, model_config, _ = get_model_from_ckpt(
                                    args.ckpt_path, 
                                    config_path=args.model_config, 
                                    print_remain=True, 
                                    ema_enabled=False, # Originally True
                                    save_name=args.save_name)
    else:
        MODEL, model_config = get_pretrained_model(args.ckpt_path)

    condition_config = model_config.get('model', None).get('conditioning', None).get('configs', None)
    
    # Pretransform
    if args.pretransform_ckpt_path != None :
        model_config_training = model_config.get('training', None)
        model_config_ema = False
        
        MODEL,_ = load_model_from_ckpt(
                MODEL, args.pretransform_ckpt_path, 
                print_remain=True, 
                ema_inference=model_config_ema, 
                only_mode="autoencoder")
        
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    sample_size_sec = sample_size//sample_rate
    print ("[Info] Sample Rate : {}, Sample Size : {}\n".format(sample_rate, sample_size))

    MODEL = MODEL.to(device).eval().requires_grad_(False)

    if args.model_half:
        MODEL.to(torch.float16)

    # Load CLAP model for in-batch similarity if enabled
    clap_model = None
    if args.in_batch_sim:
        print("[Info] Loading CLAP model for in-batch similarity calculation...")
        clap_model = load_clap_model('music_audioset_epoch_15_esc_90.14.pt')
        clap_model = clap_model.to(device).eval()
        # Force batch_size to 10 for in-batch similarity calculation
        args.batch_size = 10
        print("[Info] CLAP model loaded successfully")
        print("[Info] Batch size set to 10 for in-batch similarity calculation")

    # ========================================================= Data
    data_list = [] 
    
    # Initialize global variables
    if sample_size!=2097152: # stable-audio open
        raise Exception("[Error] Only Stable-Audio Open is supported")

    if csv_file_path.endswith('.csv'):

        with open(csv_file_path, 'r') as fi:
            reader = csv.reader(fi)
            headers = next(reader)  # Read headers
            reader_list = list(reader)

        # Find column indices from headers
        header_dict = {col.lower(): idx for idx, col in enumerate(headers)}
        
        # First column is always treated as fix_name
        fix_name_idx = 0
        
        # Find text_type column (required)
        if args.text_type == 'tag':
            text_col = 'aspect_list'
        else:
            text_col = 'caption'
        if text_col not in header_dict:
            raise Exception(f"[Error] Required column {text_col} not found in CSV headers: {headers}")

        for row in reader_list:
            seed = args.seed
            
            # Extract basic columns
            fix_name = row[fix_name_idx]  # First column is always fix_name
            

            raw_text_value = row[header_dict[text_col]]
            # If text_type is 'tag', convert list to string
            if args.text_type == 'tag':
                if isinstance(raw_text_value, (list, tuple)):
                    caption = ", ".join(map(str, raw_text_value))
                else:
                    # Try parsing as it might be a string representation of a list
                    parsed_value = None
                    try:
                        parsed_value = ast.literal_eval(raw_text_value)
                    except Exception:
                        parsed_value = None
                    if isinstance(parsed_value, (list, tuple)):
                        caption = ", ".join(map(str, parsed_value))
                    else:
                        caption = str(raw_text_value)
            else:
                caption = raw_text_value


            if 'start_s' in header_dict and 'end_s' in header_dict:
                start_sec = int(row[header_dict['start_s']])
                target_sec = int(row[header_dict['end_s']]) 
            elif 'length' in header_dict:
                start_sec = 0
                target_sec = int(row[header_dict['length']])
            else:
                start_sec = 0
                target_sec = sample_size_sec

            if start_sec >= target_sec:
                raise Exception("[Error] start_sec {} and target sec {} for {} is weird".format(start_sec, target_sec, fix_name))
            
            # Process bpm - set value if bpm column exists
            row_bpm = None
            if 'bpm' in header_dict:
                try:
                    row_bpm = int(row[header_dict['bpm']])
                except (ValueError, IndexError):
                    row_bpm = None
            
            data_list.append([fix_name, caption, start_sec, target_sec, seed, row_bpm])
                
    else :
        raise Exception("[Error] Data format must be .csv : {}".format(csv_file_path))

    print("[Info] Dataset {} is loaded".format(csv_file_path))

    ckpt_path_basename = os.path.splitext(args.ckpt_path.split('/')[-1])[0]

    new_path ='./output_inf/{}'.format(args.save_name)
    Path(new_path).mkdir(parents=True, exist_ok=True)

    # ========================================================= Start
    pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=False)
    encoding = torch.nn.Sequential(Stereo())

    # Resampler for CLAP (if needed)
    clap_sample_rate = 48000
    resampler = None
    if args.in_batch_sim and sample_rate != clap_sample_rate:
        resampler = T.Resample(sample_rate, clap_sample_rate).to(device)

    generated_files=[]
    batch_similarities = []  # Store similarity scores for each batch

    for row in data_list:        
        fix_name, prompt, start_sec, target_sec, seed, bpm = row

        
        audio_pre =None

        intersection_sec = 30 # THIS MUST BE LESS THAN SAMPLE SIZE
        intersection_size = intersection_sec*sample_rate

        sigma_min = args.sigma_min      # 0.03  50
        sigma_max = args.sigma_max     # 500   500
        if sample_size<=intersection_size:
             raise Exception("[Error] sample size({}) <= intersection size({})".format(sample_size, intersection_size))

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& MAIN CODE - MUST BE HERE
        def generation_package(start_sec_inner, target_sec_inner, mask_args_inner=None, audio_pre_inner=None):
            # ******************************************            
            conditioning_dict = {"prompt": prompt, "seconds_start": start_sec_inner, "seconds_total": target_sec_inner}
            conditioning = [conditioning_dict] * args.batch_size
            negative_conditioning = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print("\n[Info] Conditioning: {}, Seed : {}, Batch size: {}".format(conditioning[0], seed, args.batch_size))

            audio_batched_inner = generate_diffusion_cond(
                MODEL,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                batch_size=args.batch_size,
                sample_size=sample_size,
                sample_rate=sample_rate,
                seed=seed,
                device=device,

                init_audio=audio_pre_inner,
                init_noise_level = args.init_noise_level,
                mask_args = mask_args_inner,
                
                sampler_type=args.sampler_type,
                sigma_min=sigma_min,
                sigma_max=sigma_max
            )
            return audio_batched_inner
        def make_audio_pre_package(audio_batched):
            audio_pre_inner=np.empty((0,2,sample_size))
            for i in range(args.batch_size):
                audio_single=audio_batched[i].cpu().numpy() # .cpu() # [2,samplesize]
                
                # Get the last 120 seconds
                last_intersection_sec_sec = audio_single[:, -intersection_size:]

                silent_audio = np.zeros((2,sample_size-intersection_size))

                # Combine empty audio + last 300 seconds (total 6 minutes)
                final_audio = np.concatenate([last_intersection_sec_sec, silent_audio],1)  # Concatenate two arrays
                final_audio = final_audio[np.newaxis,:,:]

                audio_pre_inner = np.concatenate([audio_pre_inner,final_audio],axis=0)

                # Turn into torch tensor, converting from int16 to float32
            if audio_pre_inner.shape[0] == 0:
                raise Exception("[Error] audio_pre not concated {}".format(audio_pre_inner.shape))

            audio_pre_inner= torch.from_numpy(audio_pre_inner).float()
            audio_pre_inner = (sample_rate, audio_pre_inner)
            return audio_pre_inner

        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& 
        # current_start+intersectionsec = generating target 
        # NO NEED RECURSIVE
        if not args.recursive or target_sec - start_sec <= sample_size_sec:
            audio_batched = generation_package(start_sec,target_sec)
        # YES NEED RECURSIVE
        else:
            start_sec_iter=start_sec
            while(start_sec_iter+intersection_sec<target_sec):
                if audio_pre==None:
                    audio_batched = generation_package(start_sec_iter,target_sec)

                    start_sec_iter = start_sec_iter + int(sample_size/sample_rate) - intersection_sec

                    # ============================================init audio 
                    if start_sec_iter+intersection_sec<target_sec :
                        audio_pre = make_audio_pre_package(audio_batched)
                else:
                    # ***************************************** Mask 
                    mask_args = {
                        "cropfrom": 0.0,
                        "pastefrom": 0.0,
                        "pasteto": 100.0,
                        "maskstart": intersection_size,
                        "maskend": sample_size,
                        "softnessL": 0*sample_rate, # NOT Working
                        "softnessR": 0*sample_rate, # NOT Working
                        "marination": 0.0,
                    }
                    audio_batched_post = generation_package(start_sec_iter,target_sec, mask_args_inner=mask_args, audio_pre_inner=audio_pre)
                    start_sec_iter = start_sec_iter + int(sample_size/sample_rate) - intersection_sec

                    # =============================================concat 
                    audio_batched = torch.concat([audio_batched[:,:,:-intersection_size],audio_batched_post],dim=2)
                    
                    # ============================================init audio 
                    if start_sec_iter+intersection_sec<target_sec :
                        audio_pre = make_audio_pre_package(audio_batched)
        # ********************************************** Modified Outputd


        if audio_batched.shape[0]!=args.batch_size:
            raise Exception("[Error] Something weird with batch size {}!={}".format(audio_batched.shape[0], args.batch_size))

        # Calculate in-batch similarity if enabled
        if args.in_batch_sim:
            batch_avg_similarity = calculate_in_batch_similarity(audio_batched, clap_model, resampler, device)
            batch_similarities.append(batch_avg_similarity)
            print("[Info] In-batch similarity (CLAP) for current batch: {:.4f}".format(batch_avg_similarity))

        start_length = start_sec*sample_rate
        target_length = target_sec*sample_rate
        
        if not args.in_batch_sim or args.force_save:
            # Save all samples in the batch
            for batch_idx in range(args.batch_size):
                audio_single=audio_batched[batch_idx]

                if audio_single.shape[1]>(target_length-start_length) :
                    audio_single = audio_single[:, :target_length-start_length]

                audio_single = audio_single.to(torch.float32).div(torch.max(torch.abs(audio_single))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

                # Add batch index to filename if batch_size > 1
                if args.batch_size > 1:
                    file_name = f"{new_path}/{fix_name}_{batch_idx}"
                else:
                    file_name = f"{new_path}/{fix_name}"

                wav_file = "{}.wav".format(file_name)
                mp3_file = "{}.mp3".format(file_name)
                torchaudio.save(wav_file, audio_single, sample_rate)

                if not args.save_wav :
                    convert_wav_to_mp3(wav_file, mp3_file)
                    generated_files.append([mp3_file.split('/')[-1],prompt,start_sec, target_sec,seed])
                    os.remove(wav_file)

                    now = datetime.datetime.now()
                else : 
                    generated_files.append([wav_file.split('/')[-1], prompt,start_sec, target_sec,seed])
            if args.single:
                break
        if args.single:
            break

    if not args.in_batch_sim or args.force_save:
        fo = open("{}/{}_cfg{}_promptlist.csv".format(new_path, csv_file_path.split('/')[-1].split('.')[0], args.cfg_scale),'w')
        writer = csv.writer(fo)
        meta_schema=['File Name','Prompt', 'Length-Start', 'Length-End', 'Seed']
        writer.writerow(meta_schema)
        for row in generated_files:
            writer.writerow(row)
        fo.close()
        print("[Info] All {} files generated".format(len(generated_files)))
    
    # Print and save in-batch similarity results if enabled
    if args.in_batch_sim:
        results_txt_path = os.path.join(new_path, "in_batch_similarity.txt")
        save_in_batch_similarity_results(batch_similarities, results_txt_path)
    
