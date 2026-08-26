#!/bin/bash

#SBATCH --job-name=hemexp-nlst-eval
#SBATCH --output=hemexp-nlst-eval.out
#SBATCH --error=hemexp-nlst-eval.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# Stage 4: nodule_quality_eval on both diffusion checkpoints.
#
# --metrics-backend pyradiomics matches every other arm in this repo (their
# radiomic_features CSVs carry the full IBSI panel); the skimage backend would
# produce a different feature set and break the comparison.
#
# Runs from nodule_quality_eval/ because MedSAM2Segmenter resolves its
# checkpoint cache as the relative path ./checkpoints -- running elsewhere
# re-downloads ~750 MB.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

N=/scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval
RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs
cd "$N"

nvidia-smi --query-gpu=name --format=csv,noheader

for TAG in best final; do
  echo "=========== evaluating hemexp_${TAG} ==========="
  python run_evaluation.py \
      --input-dir "$RUNS/eval_triplets_${TAG}" \
      --input-format npy \
      --metrics-backend pyradiomics \
      --cache-dir "cache_hemexp_${TAG}" \
      --out-dir "results_hemexp_${TAG}" \
      --out-tag "hemexp_${TAG}"
done

echo "=== stage 4 complete ==="
for TAG in best final; do
  echo "--- hemexp_${TAG} ---"
  ls -la "results_hemexp_${TAG}/" 2>/dev/null
done
