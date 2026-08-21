#!/bin/bash

# CPU-only. The account's GPU allocation is held by another job (MaxGRESPerAccount),
# and this probe is small enough not to need one: ~10.4k images of 64x64 through a
# frozen VAE encoder, four times, plus a handful of logistic regressions.

#SBATCH --job-name=probe-noaug
#SBATCH --output=probe-noaug.out
#SBATCH --error=probe-noaug.err
#SBATCH --account=liyues
#SBATCH --partition=liyues-a100
#SBATCH --gpus=0
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=8

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

cd /scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/probe-noaug

python -u probe_malignancy_noaug.py \
  --reference published_none \
  --out_dir results/vae_ablation/probe_noaug
