#!/bin/bash

#SBATCH --job-name=namd3-cases
#SBATCH --output=namd3-cases.out
#SBATCH --error=namd3-cases.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

WT=/scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/namd3-sample-analysis
MAIN=/scratch/liyues_root/liyues/shxjames/NAMD
NAMD3=/scratch/liyues_root/liyues/shxjames/NLST-Img/NAMD_3

# Stage 1 (minutes): per-(seed, sample) LPIPS + ViT logits + pixel error.
echo "=========== STAGE 1: LPIPS + ViT per sample ==========="
cd "$WT"
python scripts/namd3_cases/compute_metrics.py \
    --namd_dir "$NAMD3" \
    --out_dir results/NAMD_3_cases

# Stage 2 (hours): MedSAM2 segmentation + PyRadiomics nodule properties.
# Runs from the main checkout so it shares the MedSAM2 checkpoints, and lands
# beside the other arms' results_*/cache_* dirs.
echo "=========== STAGE 2: nodule property eval ==========="
cd "$MAIN/nodule_quality_eval"
python run_evaluation.py \
    --input-dir "$NAMD3" \
    --input-format npy \
    --cache-dir cache_NAMD_3 \
    --out-dir results_NAMD_3 \
    --out-tag all

cd "$MAIN"
echo "=========== METRICS: NAMD_3 ==========="
python scripts/nodule_report.py nodule_quality_eval/results_NAMD_3 all
