import numpy as np
from tqdm import tqdm
import functools
import random
import pandas as pd
import torch
import inspect
from sklearn.kernel_approximation import Nystroem
import torchaudio
import traceback
from tqdm import tqdm
import glob
import os
from datetime import datetime
import numbers
from typing import Optional

import scipy
import scipy.linalg
from scipy.sparse.linalg import eigsh

def gaussian_kernel_decorator(function):
    def wrap_kernel(self, *args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(function)
        params = list(sig.parameters.keys())
        
        # Determine if `compute_kernel`, `algorithm`, and `sigma` parameter is in args or kwargs
        bound_args = sig.bind_partial(*args, **kwargs).arguments
        compute_kernel = bound_args.get('compute_kernel', True)
        algorithm = bound_args.get('algorithm', 'kernel')
        kernel = bound_args.get('kernel', 'gaussian')
        kernel_function = self.gaussian_kernel if kernel == 'gaussian' else self.cosine_kernel

        sigma = bound_args.get('sigma', None)
        sigma_x, sigma_y = None, None
        if type(sigma) == tuple:
            sigma_x, sigma_y == sigma[0], sigma[1]
        elif isinstance(sigma, (int, float)):
            sigma_x = sigma_y = sigma  # TODO check other types in the future
        else:
            if type(self.sigma) != tuple:
                raise ValueError(f"Self.sigma should be tuple but {type(self.sigma)} is given.")
            sigma_x, sigma_y = self.sigma

        if compute_kernel is True and algorithm == 'kernel':
            args = list(args)  # To be able to edit args
            if 'X' in params:
                index = params.index('X') - 1
                args[index] = kernel_function(args[index], sigma=sigma_x)  # TODO: this is buggy

            if 'Y' in params:
                index = params.index('Y') - 1
                if args[index] is not None:
                    args[index] = kernel_function(args[index], sigma=sigma_y)

        return function(self, *args, **kwargs)

    return wrap_kernel


def entropy_q(p, q=1, log_base='e'):
    log = torch.log if log_base == 'e' else torch.log2
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * log(p_)).sum()
    if q == "inf":
        return -log(torch.max(p))
    return log((p_ ** q).sum()) / (1 - q)


def cov_rff2(x, feature_dim, std, batchsize=8, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2  # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product)  # [B, feature_dim]
    batched_rff_sin = torch.sin(product)  # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim)  # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2)  # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx * batchsize:min((batchidx + 1) * batchsize,
                                                                 batched_rff.shape[0])]  # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()


def cov_rff2_joint(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]
    y = x.clone()
    y[:, 780:] = -y[:, 780:].clone()
    product = torch.matmul(y, omegas)
    batched_rff_cos_negative = torch.cos(product) # [B, feature_dim]
    batched_rff_sin_negative = torch.sin(product) # [B, feature_dim]


    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin, batched_rff_cos_negative, batched_rff_sin_negative], dim=1) / (np.sqrt(2) * np.sqrt(feature_dim)) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((4 * feature_dim, 4 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 4

    return cov, batched_rff.squeeze()


def cov_rff2_joint_v2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    omegas_img, omegas_txt = omegas[:780], omegas[780:]
    img, txt = x[:, :780], x[:, 780:]
    product_img = img @ omegas_img
    product_txt = txt @ omegas_txt
    batched_rff_cos = torch.cos(img @ omegas_img + txt @ omegas_txt) # [B, feature_dim]
    batched_rff_sin = torch.sin(img @ omegas_img + txt @ omegas_txt) # [B, feature_dim]
    batched_rff_cos_negative = torch.cos(img @ omegas_img - txt @ omegas_txt) # [B, feature_dim]
    batched_rff_sin_negative = torch.sin(img @ omegas_img - txt @ omegas_txt) # [B, feature_dim]

    batched_rff = torch.cat([torch.cos(product_img) * torch.cos(product_txt),
                            torch.cos(product_img) * torch.sin(product_txt),
                            torch.sin(product_img) * torch.cos(product_txt),
                            torch.sin(product_img) * torch.sin(product_txt)], dim=1) / np.sqrt(feature_dim) # [B, 4 * feature_dim]

    # batched_rff = torch.cat([batched_rff_cos, batched_rff_sin, batched_rff_cos_negative, batched_rff_sin_negative], dim=1) / (np.sqrt(2) * np.sqrt(feature_dim)) # [B, 4 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((4 * feature_dim, 4 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 4

    return cov, batched_rff.squeeze()


def cov_diff_rff(x, y, feature_dim, std, batchsize=16):
    assert len(x.shape) == len(y.shape) == 2 # [B, dim]

    B, D = x.shape
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')

    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)
    y_cov, y_feature = cov_rff2(y, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, y_cov, omegas, x_feature, y_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim], [B, 2 * feature_dim]

def cov_rff(x, feature_dim, std, batchsize=16, normalise=True):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


def joint_cov_rff(x, y, feature_dim, std_x, std_y, batchsize=16, normalise=True, omegas_x=None, omegas_y=None):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D_x = x.shape
    _, D_y = y.shape

    if omegas_x is None or omegas_y is None:
        omegas = torch.randn((D_x + D_y, feature_dim), device=x.device)
        omegas_x = omegas[:D_x] * (1 / std_x)
        omegas_y = omegas[D_x:] * (1 / std_y)
    else:
        omegas = torch.cat([omegas_x, omegas_y])

    x_cov, x_feature = cov_rff2(
        torch.cat([x, y], dim=1),
        feature_dim,
        std=None,
        batchsize=batchsize,
        presign_omeaga=torch.cat([omegas_x, omegas_y]),
        normalise=normalise
    )

    return x_cov, omegas_x, omegas_y, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]


def joint_random_projection(phi_x, phi_y, feature_dim, batchsize=16, normalise=True, u_x=None, u_y=None):
    if u_x or u_y is None:
        sqrt_3 = torch.sqrt(torch.tensor(3.0))
        u_x = (1 / torch.sqrt(torch.tensor(feature_dim))) * torch.empty((feature_dim, phi_x.shape[0]), device=phi_x.device).uniform_(-sqrt_3, sqrt_3)
        u_y = (1 / torch.sqrt(torch.tensor(feature_dim))) * torch.empty((feature_dim, phi_y.shape[0]), device=phi_y.device).uniform_(-sqrt_3, sqrt_3)

    phi_hat = (u_x @ phi_x) * (u_y @ phi_y)
    cov_hat = phi_hat @ phi_hat.T / torch.trace(phi_hat @ phi_hat.T)  # Todo check why not working with 1/n
    return cov_hat, u_x, u_y, phi_hat




"""
TruncatedVendi: A class to compute VENDI (exponentiated entropy) scores
using Gaussian (RBF) or cosine kernels, with optional Nyström approximation.
"""
class TruncatedVendi:
    """
    A class to compute VENDI (exponentiated entropy) or truncated VENDI
    scores for a given feature matrix, using Gaussian (RBF) or cosine kernels.
    Supports full/exact kernel computation or Nyström approximation.

    Attributes:
        features (torch.Tensor): Feature matrix of shape (N, D).
    """

    def __init__(self, features: torch.Tensor):
        """
        Initialize the TruncatedVendi with a feature tensor.

        Args:
            features (torch.Tensor): Tensor of shape (N, D).
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError("Features must be a 2D tensor of shape (N, D)")
        self.features = features

    def _normalized_gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Compute a normalized Gaussian (RBF) kernel matrix between x and y in batches.

        K[i, j] = exp(-||x[i] - y[j]||^2 / (2 * sigma^2)) / sqrt(N * M),
        where N = x.shape[0], M = y.shape[0].

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (M, D).
            sigma (float): Gaussian bandwidth parameter.
            batch_size (int, optional): Batch size for computing pairwise distances.

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M), normalized by sqrt(N * M).
        """
        assert x.ndim == 2 and y.ndim == 2, "Inputs must be 2D tensors"
        assert x.shape[1] == y.shape[1], "Feature dimensions must match"

        n, m = x.shape[0], y.shape[0]
        inv_coeff = -1.0 / (2.0 * sigma * sigma)
        norm_factor = (n * m) ** 0.5
        device = x.device

        kernel_batches = []
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            y_batch = y[start:end]  # shape: (b, D)

            # Compute squared Euclidean distances: (N, b)
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)           # (N, 1)
            y_norm_sq = (y_batch ** 2).sum(dim=1, keepdim=True).T    # (1, b)
            cross_term = x @ y_batch.T                               # (N, b)
            dist_sq = x_norm_sq + y_norm_sq - 2.0 * cross_term
            kernel_batch = torch.exp(inv_coeff * dist_sq)            # (N, b)
            kernel_batches.append(kernel_batch)

        K = torch.cat(kernel_batches, dim=1)  # shape: (N, M)
        return K.div_(norm_factor)

    def _cosine_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a normalized cosine similarity kernel between x and y.

        K[i, j] = cosine_similarity(x[i], y[j]) / sqrt(N * M).

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (M, D).

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        assert x.ndim == 2 and y.ndim == 2, "Inputs must be 2D tensors"

        # Normalize each vector to unit norm along dim=1
        x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)

        sim_matrix = x_norm @ y_norm.T  # shape: (N, M)
        n, m = x.shape[0], y.shape[0]
        return sim_matrix.div_((n * m) ** 0.5)

    def _nystrom_kernel(
        self,
        x_feats: torch.Tensor,
        kernel_name: str,
        n_components: int,
        sigma: Optional[float] = None,
        random_state: int = 1,
    ) -> torch.Tensor:
        """
        Compute an approximate kernel matrix via the Nyström method.

        Supports 'gaussian' (RBF) or 'cosine'.

        Args:
            x_feats (torch.Tensor): Data points shape (N, D). Must be convertible to numpy.
            kernel_name (str): One of ['gaussian', 'cosine'].
            n_components (int): Number of landmark points (<= N).
            sigma (float, optional): Bandwidth for Gaussian kernel.
            random_state (int, optional): Random seed for Nyström sampling.

        Returns:
            torch.Tensor: Approximated kernel matrix of shape (n_components, n_components),
                          normalized by dividing by N.
        """
        assert kernel_name in ("gaussian", "cosine"), "kernel_name must be 'gaussian' or 'cosine'"
        N, _ = x_feats.shape
        n_components = min(n_components, N)

        if kernel_name == "gaussian":
            assert isinstance(sigma, numbers.Number), "sigma must be provided for Gaussian kernel"
            gamma = 1.0 / (2.0 * sigma * sigma)  # sklearn expects gamma = 1/(2σ²)
            sklearn_kernel = "rbf"
        else:
            gamma = None
            sklearn_kernel = "cosine"

        # Convert to numpy for sklearn
        X_np = x_feats.cpu().numpy()
        nystroem = Nystroem(
            kernel=sklearn_kernel,
            gamma=gamma,
            n_components=n_components,
            random_state=random_state,
        )
        transformed = nystroem.fit_transform(X_np)  # shape: (N, n_components)

        # Compute low-rank approximation: (n_components, n_components)
        K_approx = transformed.T @ transformed  # numpy array
        K_approx = K_approx / float(N)
        return torch.from_numpy(K_approx)

    def _calculate_stats(
        self,
        eigenvalues: torch.Tensor,
        alpha: float = 2.0,
        truncation: Optional[int] = None,
    ) -> float:
        """
        Compute VENDI (exponentiated entropy) or truncated VENDI from eigenvalues.

        If alpha == 1: Shannon entropy; otherwise, Rényi entropy of order alpha.

        If truncation is provided and < len(eigenvalues), we add the tail mass uniformly.

        Args:
            eigenvalues (torch.Tensor): 1D tensor of eigenvalues (must be >= 0).
            alpha (float, optional): Order of Rényi entropy; alpha=1 uses Shannon.
            truncation (int, optional): Number of top eigenvalues to keep (t < len).

        Returns:
            float: Rounded VENDI score (to two decimals).
        """
        eps = 1e-10
        ev = eigenvalues.clamp(min=eps)
        ev, _ = ev.sort(descending=True)

        if isinstance(truncation, int) and 0 < truncation < ev.numel():
            top = ev[:truncation]
            tail_mass = 1.0 - top.sum()
            top = top + (tail_mass / truncation)
            ev = top

        log_ev = ev.log()
        if abs(alpha - 1.0) < 1e-8:
            # Shannon entropy
            entropy = - (ev * log_ev).sum()
        else:
            # Rényi entropy: (1 / (1 - alpha)) * log(sum(ev^alpha))
            entropy = (1.0 / (1.0 - alpha)) * (ev.pow(alpha).sum().log())

        vendi = torch.exp(entropy)
        return vendi.item()

    def compute_score(
        self,
        alpha: float,
        truncation: Optional[int] = None,
        sigma: Optional[float] = None,
        kernel: str = "gaussian",
        use_nystrom: bool = False,
        batch_size: int = 64,
    ) -> float:
        """
        Compute the VENDI or truncated VENDI score for the stored feature matrix,
        using the specified kernel and parameters.

        Args:
            alpha (float): Entropy order (1 for Shannon, >1 for Rényi).
            truncation (int, optional): Number of top eigenvalues to keep.
            sigma (float, optional): Bandwidth for Gaussian kernel.
            kernel (str, optional): One of ['gaussian', 'cosine'].
            use_nystrom (bool, optional): If True, use Nyström approximation.
            batch_size (int, optional): Batch size for computing Gaussian kernel.

        Returns:
            float: VENDI score (rounded to two decimals).
        """
        assert kernel in ("gaussian", "cosine"), "kernel must be 'gaussian' or 'cosine'"

        if use_nystrom and isinstance(truncation, int):
            K = self._nystrom_kernel(
                x_feats=self.features,
                kernel_name=kernel,
                n_components=truncation,
                sigma=sigma,
            )
        else:
            if kernel == "gaussian":
                assert isinstance(sigma, numbers.Number), "sigma must be provided for Gaussian kernel"
                K = self._normalized_gaussian_kernel(
                    self.features, self.features, sigma, batch_size
                )
            else:  # "cosine"
                K = self._cosine_kernel(self.features, self.features)

        # Ensure symmetry / numerical stability
        K = (K + K.T) / 2.0
        eigenvals = torch.linalg.eigvalsh(K)
        return self._calculate_stats(eigenvals, alpha=alpha, truncation=truncation)



