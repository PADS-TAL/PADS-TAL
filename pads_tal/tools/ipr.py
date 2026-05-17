#!/usr/bin/env python3
import os
from collections import namedtuple
import numpy as np

import typing as tp
from pathlib import Path
from tqdm import trange
import torch

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisionAndRecall', ['precision', 'recall'])

def get_ipr_info(ipr_basedata, ipr_basemodel, num_avg=1, generation_target:tp.Literal['','songdesc','songdescriber','melbench']='', dataset_genre=''):
    if ipr_basedata=="songdesc" or ipr_basedata=="songdescriber":
        if ipr_basemodel=="music":
            ipr_basemodel="clap-f"

        if ipr_basemodel=="clap-f":
            ipr_mode="audio2"
            ipr_fixed_dim=-1
            ipr_ref_path = "/Path/To/Reference"
            num_avg=1
        else:
            raise Exception("[IPR] for songdescriber : [clap-f] supported")
    elif ipr_basedata=="melbench":
        # Recommend : clap-f, clap-fa
        if ipr_basemodel=="music":
            ipr_basemodel="clap-f"
        
        if ipr_basemodel=="clap-f":
            # [53217, 512]
            ipr_mode="audio2"
            ipr_fixed_dim=-1
            if dataset_genre:
                if dataset_genre == 'elec':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'jazz':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'pop':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'newage':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'blues':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'classic':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'country':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'latin':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'metal':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'rock':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'easy':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'folk':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'rnb':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'hiphop':
                    ipr_ref_path = "/Path/To/Reference"
                elif dataset_genre == 'world':
                    ipr_ref_path = "/Path/To/Reference"
                else:
                    raise Exception("[IPR][Error] melBench dataset for genre {} is not supported".format(dataset_genre))
            else:
                ipr_ref_path = "/Path/To/Reference"

            num_avg=1
        else:
            raise Exception("[IPR] for melBench : [clap-f] supported")

    else:
        raise Exception("[IPR] weird basedata : {} [Available : songdesc(riber), melbench]".format(ipr_basedata))
    if not os.path.isfile(ipr_ref_path):
        raise Exception("Invalid Path : {}".format(ipr_ref_path))

    # **************************************

    if ipr_basemodel in ['clap-f']:
        triple_reference = True
        triple_generation=True
        random_generation=False
    else:
        triple_reference = False
        triple_generation=False
        random_generation=True

    reference_options ={
        'model':ipr_mode,
        'path_real':ipr_ref_path,
        'fixed_dim':ipr_fixed_dim,
        'triple_reference' :triple_reference
    }
    generation_options={
        'fixed_dim':ipr_fixed_dim,
        'triple_generation': triple_generation,
        'random_generation': random_generation
    }
    return reference_options, generation_options, num_avg
#========================================================IPR
class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=-1, model="", path_real=None, fixed_dim=-1, triple_reference=False):
        self.manifold_ref = None
        self.k = k
        self.model=model
        if self.model=="audio2":
            from .ipr_model import AudioExtractor2
            self.extractor=AudioExtractor2(batch_size=batch_size, num_samples=num_samples,content_type='music')
            self.random_generation=True
        elif self.model=="audio3":
            from .ipr_model import AudioExtractor2
            self.extractor=AudioExtractor2(batch_size=batch_size, num_samples=num_samples,content_type='audio')
        else:
            raise Exception("Wrong mode : {}".format(self.model))
            
        if path_real==None:
            raise Exception("Give Reference Data path")
        self.triple_reference=triple_reference
        if not triple_reference:
            self.manifold_ref = self.compute_manifold(path_real, fixed_dim=fixed_dim)
        else :
            self.manifold_ref = self.compute_manifold(path_real, fixed_dim=fixed_dim)
            p = Path(path_real)
            paths = [str(p.with_name(f"{p.stem}{i}{p.suffix}")) for i in (1, 2, 3)]
            self.manifold_ref1 = self.compute_manifold(paths[0], fixed_dim=fixed_dim)
            self.manifold_ref2 = self.compute_manifold(paths[1], fixed_dim=fixed_dim)
            self.manifold_ref3 = self.compute_manifold(paths[2], fixed_dim=fixed_dim)

    def __call__(self, subject, fixed_dim=-1, triple_generation:bool=False, random_generation:bool=True):
        return self.precision_and_recall_with(subject, fixed_dim=fixed_dim, triple_generation=triple_generation, random_generation=random_generation)

    def precision_and_recall_with(self, subject, fixed_dim=-1, triple_generation:bool=False, random_generation:bool=True):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.Step1_compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert (self.manifold_ref is not None), "call IPR.Step1_compute_manifold_ref() first"
        if self.triple_reference:
            assert (self.manifold_ref1 is not None), "call IPR.Step1_compute_manifold_ref() first"


        if triple_generation:
            if self.triple_reference:
                manifold_subject, manifold_subject1, manifold_subject2,  manifold_subject3 = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=True, return_seg=True)

                precision1 = compute_metric(self.manifold_ref1, manifold_subject.features, 'computing precision 15% ref...')
                precision2 = compute_metric(self.manifold_ref2, manifold_subject.features, 'computing precision 50% ref...')
                precision3 = compute_metric(self.manifold_ref3, manifold_subject.features, 'computing precision 85% ref...')
                precision = np.mean([precision1, precision2, precision3])

                recall1 = compute_metric(manifold_subject1, self.manifold_ref.features, 'computing recall for 15%...')
                recall2 = compute_metric(manifold_subject2, self.manifold_ref.features, 'computing recall for 50%...')
                recall3 = compute_metric(manifold_subject3, self.manifold_ref.features, 'computing recall for 85%...')
                recall = np.mean([recall1, recall2, recall3])
            else:
                manifold_subject, manifold_subject1, manifold_subject2,  manifold_subject3 = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=True, return_seg=True)
                precision1 = compute_metric(self.manifold_ref, manifold_subject1.features, 'computing precision for 15%...')
                precision2 = compute_metric(self.manifold_ref, manifold_subject2.features, 'computing precision for 50%...')
                precision3 = compute_metric(self.manifold_ref, manifold_subject3.features, 'computing precision for 85%...')
                precision = np.mean([precision1, precision2, precision3])
    
                # fix to 15%
                recall = compute_metric(manifold_subject2, self.manifold_ref.features, 'computing recall ...')
        else:
            if self.triple_reference:
                if random_generation:
                    manifold_subject = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=False)
                else:
                    # Fix to 15% (CLAP)
                    manifold_subject = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=True, force_single=1)

                precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision ...')
                recall1 = compute_metric(manifold_subject, self.manifold_ref1.features, 'computing recall 15% ref...')
                recall2 = compute_metric(manifold_subject, self.manifold_ref2.features, 'computing recall 50% ref...')
                recall3 = compute_metric(manifold_subject, self.manifold_ref3.features, 'computing recall 85% ref...')

                recall = np.mean([recall1, recall2, recall3])
            else:
                if random_generation:
                    manifold_subject = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=False)
                else:
                    # Fix to 15% (CLAP)
                    manifold_subject = self.compute_manifold(subject, fixed_dim=fixed_dim, use_seg=True, force_single=1)
                precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
                recall = compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')

        return PrecisionAndRecall(precision, recall)

    def compute_manifold(self, input, fixed_dim=-1, use_seg=False, force_single:int=0, return_seg=False):
        '''
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        '''
        # features
        if isinstance(input, str):
            if input.endswith('.npz'):  # input is precalculated file
                print('Calculating', input)
                f = np.load(input)
                feats = f['feature']
                radii = f['radii']
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self.extractor.extract_features_from_files(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
        elif isinstance(input, torch.Tensor):
            feats = self.extractor.extract_features_single(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extractor.extract_features_single(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extractor.extract_features_single(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extractor.extract_features_single(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extractor.extract_features_from_files(input, fixed_dim=fixed_dim, use_seg=use_seg, force_single=force_single)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError
        #==================================================

        # radii
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        if use_seg and return_seg:
            feats1 = feats[0::3]
            feats2 = feats[1::3]
            feats3 = feats[2::3]
            distances1 = compute_pairwise_distances(feats1)
            radii1 = distances2radii(distances1, k=self.k)
            distances2 = compute_pairwise_distances(feats2)
            radii2 = distances2radii(distances2, k=self.k)
            distances3 = compute_pairwise_distances(feats3)
            radii3 = distances2radii(distances3, k=self.k)

            return (
                Manifold(feats, radii),
                Manifold(feats1, radii1),
                Manifold(feats2, radii2),
                Manifold(feats3, radii3)
            )

        else:
            return Manifold(feats, radii)

    def save_ref(self, fname):
        print('saving manifold to ', fname, '...')
        np.savez_compressed(fname,
                            feature=self.manifold_ref.features,
                            radii=self.manifold_ref.radii)

#========================================================functions

def compute_pairwise_distances(X, Y=None):
    '''
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    '''
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
    else:
        num_Y = Y.shape[0]
    X = X.astype(np.float64)  # to prevent underflow
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)


    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)


    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)


    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    diff_square = X_square - 2*XY + Y_square

    # check negative distance **************************
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
              min_diff_square)
    #****************************************************

    distances = np.sqrt(diff_square)
    return distances


def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


# ===================================================================
def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in trange(num_subjects, desc=desc):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects
# ===================================================================

