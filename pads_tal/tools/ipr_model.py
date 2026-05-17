import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.models as models
import tools.laion_clap_f as laion_clap
from clap_module.factory import load_state_dict
import openl3
import librosa
import pyloudnorm as pyln
import soxr
from tqdm import tqdm, trange
import requests

class AudioFolder(Dataset):
    def __init__(self, root,transform=None, recursive_bool=False):
        self.fnames = glob(os.path.join(root, '*.mp3'), recursive=recursive_bool) + \
            glob(os.path.join(root, '*.wav'), recursive=recursive_bool) 
        if len(self.fnames) == 0:
            raise ValueError('No files with this extension in this path! : {}'.format(root))    
        self.transform = transform

    def __getitem__(self, index):
        audio_path = self.fnames[index]
        return audio_path

    def __len__(self):
        return len(self.fnames)

def get_custom_loader(dir_or_fnames, batch_size=50, num_workers=4, num_samples=-1):

    # ******************************************
    # ******************************************

    if isinstance(dir_or_fnames, list):
        dataset = FileNames(dir_or_fnames)
    elif isinstance(dir_or_fnames, str):
        dataset = AudioFolder(dir_or_fnames)
    else:
        raise TypeError

    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples]
    # ******************************************
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    # ******************************************
    return data_loader

class AudioExtractor2():
    def __init__(self, batch_size=50, num_samples=-1, model=None, content_type='audio', samplingrate=48000):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.samplingrate=samplingrate
        if model is None:
            print('loading CLAP for improved precision and recall...', end='', flush=True)

            # load model
            if content_type=='music':
                url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
                clap_path = './weight/laion_clap/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt'
                model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
            elif content_type=='audio':
                url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
                clap_path = './weight/laion_clap/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/630k-audioset-fusion-best.pt'
                model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
            else: 
                raise Exception("[IPR_MODEL][Error] content type must be music or audio but {}".format(content_type))

            # download clap_model if not already downloaded
            if not os.path.exists(clap_path):
                print('Downloading ', clap_path, '...')
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
            model.model.load_state_dict(pkg)
            self.clap=model
            print('Loaded done')
            
        else:
            self.clap = model
        self.clap = self.clap.cuda().eval().requires_grad_(False)
        self.mode="audio2"

    def extract_features_single(self, audios, print_t=True, fixed_dim=-1, use_seg=False, force_single:int=0):
        desc = 'extracting features of %d audios' % len(audios)
        num_batches = int(np.ceil(len(audios) / self.batch_size))
        features = []
        if print_t:
            range_t = trange(num_batches, desc=desc)
        else:
            range_t = range(num_batches)

        for bi in range_t:
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = audios[start:end]
            audios=[]
            for audio_path in batch:
                audio, _ = librosa.load(audio_path, sr=self.samplingrate, mono=True) # sample rate should be 48000
                audio = pyln.normalize.peak(audio, -1.0)
                audio = audio.reshape(1, -1) # unsqueeze (1,T)
                audio = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(audio))).float()
                audio_embeddings = self.clap.get_audio_embedding_from_data(x = audio, use_tensor=True, use_seg=use_seg, force_single=force_single)
                audios.append(audio_embeddings.cpu().numpy())
            embeddings = np.concatenate(audios, axis=0)
            features.append(embeddings)

        return np.concatenate(features, axis=0)


    def extract_features_from_files(self, path_or_fnames, fixed_dim=-1, use_seg=False, force_single:int=0):
        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)

        num_found_audios = len(dataloader.dataset)
        desc = 'extracting features of %d audios' % num_found_audios
        if self.num_samples>0 and (num_found_audios < self.num_samples):
            print('WARNING: num_found_audios(%d) < num_samples(%d)' % (num_found_audios, self.num_samples))
        features = []
        for batch in tqdm(dataloader, desc=desc):
            audios=[]
            for audio_path in batch:
                audio, _ = librosa.load(audio_path, sr=self.samplingrate, mono=True) # sample rate should be 48000
                audio = pyln.normalize.peak(audio, -1.0)
                audio = audio.reshape(1, -1) # unsqueeze (1,T)
                audio = torch.from_numpy(self.int16_to_float32(self.float32_to_int16(audio))).float()
                audio_embeddings = self.clap.get_audio_embedding_from_data(x = audio, use_tensor=True, use_seg=use_seg, force_single=force_single)
                audios.append(audio_embeddings.cpu().numpy())
            embeddings = np.concatenate(audios, axis=0)
            features.append(embeddings)
        result = np.concatenate(features, axis=0)
        return result
        return np.concatenate(features, axis=0)
    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

