#!/bin/bash

#SBATCH --job-name=abl-none-sf
#SBATCH --output=abl-none-sf.out
#SBATCH --error=abl-none-sf.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 40GB OOM-killed a previous eval_autoencoder run (MaxRSS 41.9GB): it holds the
# model plus the checkpoint on top of the augmentation pool. 60GB has headroom.
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=4

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

cd /scratch/liyues_root/liyues/shxjames/NAMD-abl0

# Cross-check the scale_factor for the NONE-arm (0/0) VAE.
#
# run_abl_none_llm.sh pins scale_factor=0.395414, taken from the only prior config
# that used this exact VAE (np_lr2e-5_bs8_4bit_original). scale_factor is
# config-only -- no checkpoint load can catch a mismatch -- so verify rather than
# trust. Expect "Recommended scale_factor" near 0.3954; a materially different
# number means the training run must be restarted with the corrected value.
python eval_autoencoder.py \
    --config configs/autoencoder/ablation-VAE.yaml \
    --ckpt checkpoints/autoencoder/v1.5-autoencoder-original-0211.ckpt
