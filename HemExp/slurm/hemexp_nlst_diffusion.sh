#!/bin/bash

#SBATCH --job-name=hemexp-nlst-diff
#SBATCH --output=hemexp-nlst-diff.out
#SBATCH --error=hemexp-nlst-diff.err
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

# Stage 2/2: latent statistics for the stage-1 autoencoder, then the conditional
# latent diffusion. 91 steps/epoch (balanced sampler = 2 x 725), 150 epochs.
# Submit with a dependency on stage 1:
#   sbatch --dependency=afterok:<ae_jobid> slurm/hemexp_nlst_diffusion.sh

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
export WANDB_PROJECT=HEMEXP-NLST

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

cd /scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/hemexp-nlst/HemExp

RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs
AE_CKPT=$RUNS/autoencoder/checkpoints/checkpoint_best.pth

if [ ! -f "$AE_CKPT" ]; then
  echo "FATAL: stage-1 checkpoint missing: $AE_CKPT" >&2
  exit 1
fi

nvidia-smi

# Whitening statistics are AE-specific; training with the identity placeholders
# would mis-scale every latent, so this must run before the diffusion stage.
echo "=== computing latent statistics ==="
python compute_latent_stats.py \
    --ae-ckpt "$AE_CKPT" \
    --out "$RUNS/latent_stats.json" \
    --update-config CONFIGS/config_nlst.json

echo "=== training diffusion ==="
python train_diffusion.py CONFIGS/config_nlst.json

echo "=== stage 2 complete ==="
ls -la "$RUNS"/diffusion/
