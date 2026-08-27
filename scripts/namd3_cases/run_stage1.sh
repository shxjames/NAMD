#!/bin/bash

# Stage 1 only (per-sample LPIPS + ViT logits). Split out from run_metrics.sh so
# it can run alongside the hours-long nodule eval rather than behind it.

#SBATCH --job-name=namd3-stage1
#SBATCH --output=namd3-stage1.out
#SBATCH --error=namd3-stage1.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-task=8

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

cd /scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/namd3-sample-analysis
python scripts/namd3_cases/compute_metrics.py \
    --namd_dir /scratch/liyues_root/liyues/shxjames/NLST-Img/NAMD_3 \
    --out_dir results/NAMD_3_cases
