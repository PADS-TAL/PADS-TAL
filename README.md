# PADS-TAL: Padding-Annealed Diffusion Sampling in Text-Aware Latent Space for Robust and Diverse Text-to-Music Generation

Official codebase for the paper **"PADS-TAL: Padding-Annealed Diffusion Sampling in Text-Aware Latent Space for Robust and Diverse Text-to-Music Generation"**.

> ****Paper****: https://openreview.net/forum?id=c0iisI5tJj
> ****Poster****: https://icml.cc/virtual/2026/poster/62907
> ****Project page / samples****: https://pads-tal.github.io/PADS-TAL.io

## Overview

This repository provides the training, inference, and evaluation code for **PADS-TAL**, a unified pipeline for robust and diverse text-to-music generation.

- **PADS**: perturbs only the padding-indexed subspace during sampling for controlled exploration with reduced semantic drift.
- **TAL**: constructs a text-aware latent space that better preserves genre-consistent diversity.

## Repository Structure

```text
project_root/
├── config/
├── input_inf/
├── pads_tal/
├── requirements.txt
└── README.md
```

### Main entry points

- `pads_tal/train.py`: training for TAL and the diffusion model
- `pads_tal/inference.py`: sampling with PADS or CADS
- `pads_tal/eval.py`: evaluation with selected metrics

## Installation

Tested on **Ubuntu 22.04**, **CUDA 11.8**, **Python 3.10.19**, and **PyTorch 2.7.1**.

```bash
pip3 install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1
pip3 install -r requirements.txt
```

For evaluation, additionally clone Stable Audio Metrics:

```bash
git clone https://github.com/Stability-AI/stable-audio-metrics.git \
          pads_tal/tools/stable-audio-metrics
```

## Training

### 1) Train TAL (MoE-mVAE)

```bash
python3 ./pads_tal/train.py \
    --dataset-config ./config/sample_dataset.json \
    --model-config ./config/tal_pads/tal_vae.json \
    --pretrained-ckpt-path ./Path/To/checkpoint.ckpt \
    --batch-size 4 \
    --num-gpus 8 \
    --num-workers 4 \
    --wandb-skip \
    --name mvae_train_1
```

### 2) Train diffusion model in TAL space

```bash
python3 ./pads_tal/train.py \
    --dataset-config ./config/sample_dataset.json \
    --model-config ./config/tal_pads/tal_dm.json \
    --pretransform-ckpt-path ./Path/To/checkpoint.ckpt \
    --batch-size 4 \
    --num-gpus 8 \
    --num-workers 4 \
    --wandb-skip \
    --name mvae_dm_train_1
```

## Inference

### PADS

```bash
python3 ./pads_tal/inference.py \
    --ckpt-path /Path/To/checkpoint.ckpt \
    --model-config ./config/tal_pads/tal_dm_pads.json \
    --dataset songdescriber \
    --text-type tag \
    --save-name test_single \
    --save-wav
```

### CADS

```bash
python3 ./pads_tal/inference.py \
    --ckpt-path /Path/To/checkpoint.ckpt \
    --model-config ./config/tal_pads/tal_dm_cads.json \
    --dataset songdescriber \
    --text-type tag \
    --save-name test_single \
    --save-wav
```

## Evaluation

```bash
python3 ./pads_tal/eval.py \
    --result-path evaluated \
    --root-path output_inf/T1/test_single \
    --modes ipr \
    --ext .wav \
    --clap-basemodel music \
    --dataset songdescriber \
    --text-type tag \
    --ipr-basedata songdescriber
```

Available modes for `--modes`:

- `clap`
- `kld`
- `fd`
- `ipr`
- `vendi`

> **Note**  
> IPR evaluation requires precomputed reference embeddings. Please generate the reference files with `save_ref()` in `ipr.py` and update `ipr_ref_path` before evaluation.

## Notes

- Training and sampling are controlled through JSON config files in `config/`.
- Input benchmark CSVs for inference and evaluation are expected under `input_inf/eval/`.
- Checkpoints in `.ckpt` and `.safetensors` formats are supported for inference.

## Citation

If you find this repository useful, please cite the paper.

```bibtex
@inproceedings{
anonymous2026padstal,
title={{PADS}-{TAL}: Padding-Annealed Diffusion Sampling in Text-Aware Latent Space for Robust and Diverse Text-to-Music Generation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=c0iisI5tJj}
}
```
