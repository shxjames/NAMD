#!/bin/bash

#SBATCH --job-name=hemexp-nlst-ae
#SBATCH --output=hemexp-nlst-ae.out
#SBATCH --error=hemexp-nlst-ae.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# Stage 1/2: HemExp MultiHeadAEKL first stage on NLST nodule patches.
# 1800 images, 112 steps/epoch at batch 16, 150 epochs = 16800 steps.
# Writes checkpoint_best.pth (lowest val loss), which stage 2 consumes.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
export WANDB_PROJECT=hemexp-nlst-autoencoder

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

cd /scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/hemexp-nlst/HemExp

nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python train_autoencoder_nlst.py --config CONFIGS/config_ae_nlst.json

echo "=== stage 1 complete ==="
ls -la /scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs/autoencoder/checkpoints/
