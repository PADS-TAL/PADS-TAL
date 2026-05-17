
import os
import requests
from tqdm import tqdm
import torch
import numpy as np

import tools.laion_clap_f as laion_clap

from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln
import csv
import argparse
import torchaudio
import pandas as pd
from glob import glob

from torch.utils.data import DataLoader
from setproctitle import *
from stable_audio_tools.data.utils import PadCrop_Normalized_T

############################################
### CLAP functions
############################################
# following documentation from https://github.com/LAION-AI/CLAP
def _int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def _float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def _load_model(clap_model='music_audioset_epoch_15_esc_90.14.pt'):
    if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        clap_path = './weight/laion_clap/clap_model_name/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        clap_path = './weight/laion_clap/clap_model_name/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        clap_path = './weight/laion_clap/clap_model_name/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        clap_path = './weight/laion_clap/clap_model_name/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    else:
        raise ValueError('clap_model not implemented')
    
    # download clap_model if not already downloaded
    if not os.path.exists(clap_path):
        print('Downloading ', clap_model, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))
                    
    # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg, strict=False)
    model.eval()
    return model

def _get_audio_embed(audio_path, model, use_seg=False):
    with torch.no_grad():
        audio, _ = librosa.load(audio_path, sr=48000, mono=True) # sample rate should be 48000
        audio = pyln.normalize.peak(audio, -1.0)
        audio = audio.reshape(1, -1) # unsqueeze (1,T)
        audio = torch.from_numpy(_int16_to_float32(_float32_to_int16(audio))).float()
        audio_embeddings = model.get_audio_embedding_from_data(x = audio, use_tensor=True, use_seg=use_seg)
    return audio_embeddings

def _get_cosine_sim(audio_embeddings, text_emb, use_seg=False):
    if use_seg:
        cosine_sims = torch.nn.functional.cosine_similarity(audio_embeddings, text_emb.unsqueeze(0), dim=1, eps=1e-8)
        return cosine_sims.mean()
    else:
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_emb.unsqueeze(0), dim=1, eps=1e-8)[0] # text_emb: 512 -> (1,512)
        return cosine_sim

def clap_score(id2text, audio_path, audio_files_extension='.wav', clap_model='630k-audioset-fusion-best.pt', filename_mapping=None, use_seg=False):
    model = _load_model(clap_model)

    if not os.path.isdir(audio_path):        
        raise ValueError('audio_path does not exist')

    if id2text:   
        print('[EXTRACTING CLAP TEXT EMBEDDINGS] ')
        batch_size = 64
        text_emb = {}
        for i in tqdm(range(0, len(id2text), batch_size)):
            batch_ids = list(id2text.keys())[i:i+batch_size]
            batch_texts = [id2text[id] for id in batch_ids]
            with torch.no_grad():
                embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
            for id, emb in zip(batch_ids, embeddings):
                text_emb[id] = emb
    else:
        raise ValueError('Must specify id2text')

    print('[EVALUATING CLAP] ', audio_path)
    score = 0
    count = 0
    score_dict = {}
    for id in tqdm(id2text.keys()):
        # Use mapped filename if available, otherwise use original id
        if filename_mapping:
            actual_filename = filename_mapping.get(id, id)
            file_path = os.path.join(audio_path, actual_filename)
        else:
            actual_filename = id
            file_path = os.path.join(audio_path, str(id)+audio_files_extension)
        
        audio_embeddings = _get_audio_embed(file_path, model, use_seg=use_seg)
        cosine_sim = _get_cosine_sim(audio_embeddings,  text_emb[id], use_seg=use_seg)
        
        score += cosine_sim
        count += 1
        score_dict[id] = cosine_sim.item()

    return score / count if count > 0 else 0, score_dict


############################################
### In-Batch Similarity functions
############################################
def extract_clap_embeddings(audio_batch, clap_model, resampler=None, device='cuda'):
    """
    Extract CLAP embeddings from audio batch tensor.
    Args:
        audio_batch: [batch_size, channels, time] tensor
        clap_model: CLAP model
        resampler: Resampler transform (None if not needed)
        device: Device to use for computation
    Returns:
        embeddings: [batch_size, embedding_dim] tensor
    """
    batch_size = audio_batch.shape[0]
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(batch_size):
            audio_single = audio_batch[i]  # [channels, time]
            
            # Convert to mono if stereo
            if audio_single.shape[0] > 1:
                audio_single = audio_single.mean(dim=0, keepdim=True)  # [1, time]
            
            # Resample if needed
            if resampler is not None:
                audio_single = resampler(audio_single)  # [1, resampled_time]
            
            # Convert to numpy and normalize
            audio_np = audio_single.squeeze(0).cpu().numpy()  # [time]
            
            # Normalize using pyloudnorm (peak normalization to -1.0)
            audio_np = pyln.normalize.peak(audio_np, -1.0)
            
            # Convert to int16 and back to float32 (as done in CLAP preprocessing)
            audio_int16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
            audio_float32 = (audio_int16 / 32767.0).astype(np.float32)
            
            # Reshape to [1, time] for CLAP
            audio_float32 = audio_float32.reshape(1, -1)
            audio_tensor = torch.from_numpy(audio_float32).float().to(device)
            
            # Extract embedding
            audio_embedding = clap_model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True, use_seg=True)
            embeddings_list.append(audio_embedding)
    
    # Stack all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)  # [batch_size, embedding_dim]
    return embeddings

def calculate_in_batch_similarity(audio_batch, clap_model, resampler=None, device='cuda'):
    """
    Calculate in-batch pairwise cosine similarity using CLAP embeddings.
    Args:
        audio_batch: [batch_size, channels, time] tensor
        clap_model: CLAP model
        resampler: Resampler transform (None if not needed)
        device: Device to use for computation
    Returns:
        batch_avg_similarity: Average pairwise cosine similarity for this batch
    """
    batch_size = audio_batch.shape[0]
    
    # Extract CLAP embeddings from audio batch
    audio_embeddings = extract_clap_embeddings(audio_batch, clap_model, resampler, device)
    
    # Normalize embeddings for cosine similarity
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
    
    # Calculate pairwise cosine similarities
    # Using matrix multiplication: embeddings @ embeddings.T gives pairwise similarities
    similarity_matrix = torch.mm(audio_embeddings, audio_embeddings.T)  # [batch_size, batch_size]
    
    triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=similarity_matrix.device)
    pairwise_similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
    
    # Calculate average pairwise similarity for this batch
    batch_avg_similarity = pairwise_similarities.mean().item()
    
    return batch_avg_similarity

def save_in_batch_similarity_results(batch_similarities, save_path):
    """
    Save in-batch similarity results to a text file.
    Args:
        batch_similarities: List of similarity scores for each batch
        save_path: Path to save the results file
    """
    if len(batch_similarities) > 0:
        overall_avg_similarity = np.mean(batch_similarities)
        print("\n[Info] ========================================")
        print("[Info] In-Batch Similarity Results (CLAP embeddings):")
        print("[Info]   Number of batches: {}".format(len(batch_similarities)))
        print("[Info]   Average pairwise cosine similarity: {:.4f}".format(overall_avg_similarity))
        print("[Info] ========================================\n")
        
        with open(save_path, "w") as f:
            f.write("In-Batch Similarity Results (CLAP embeddings)\n")
            f.write("Number of batches: {}\n".format(len(batch_similarities)))
            for idx, sim in enumerate(batch_similarities, start=1):
                f.write("Batch {}: avg_cosine={:.4f}\n".format(idx, sim))
            f.write("Average pairwise cosine similarity: {:.4f}\n".format(overall_avg_similarity))
        
        return overall_avg_similarity
    else:
        print("[Warning] In-batch similarity was enabled but no batches were processed")
        return None

############################################
### Extra functions
############################################
def get_actual_fname(gen_audio_path, gen_prompt_csv):
    df = pd.read_csv(os.path.join(gen_audio_path, gen_prompt_csv))

    id2text = df.set_index('File Name')['Prompt'].to_dict()
    
    actual_files = os.listdir(gen_audio_path)
    actual_files = [f for f in actual_files if f.endswith('.mp3')]
    filename_mapping = {}
    for csv_filename in df['File Name']:
        actual_filename = _map_csv_filename_to_actual_filename(csv_filename, actual_files)
        filename_mapping[csv_filename] = actual_filename
        # print(f"CSV: {csv_filename} -> Actual: {actual_filename}")
    return df, id2text, filename_mapping

def _map_csv_filename_to_actual_filename(csv_filename, actual_files_list):
    """
    Map CSV filename format to actual filename format.
    CSV format: "Jazz_keyboard_hammond organ_keyboard (musical)_110_B0.mp3"
    Actual format: "Jazz_keyboard_hammond_organ_keyboard_(musical)_110_B0.mp3"
    """
    mapped_name = csv_filename.replace(" ", "_").replace("(", "(").replace(")", ")")
    
    if mapped_name in actual_files_list:
        return mapped_name
    
    for actual_file in actual_files_list:
        csv_base = csv_filename.replace(".mp3", "")
        actual_base = actual_file.replace(".mp3", "")
        
        csv_clean = csv_base.replace(" ", "_").replace("(", "(").replace(")", ")")
        actual_clean = actual_base
        
        if csv_clean == actual_clean:
            return actual_file
    return csv_filename
