#!/bin/bash

#SBATCH --job-name=hemexp-pregen-eval
#SBATCH --output=hemexp-pregen-eval.out
#SBATCH --error=hemexp-pregen-eval.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# Stage 5: the metrics nodule_quality_eval does NOT cover -- ViT AUROC/AUPRC,
# LPIPS, FID, HaarPSI, and the global / nodule / background MSE split.
#
# Uses NLST-Img/eval_pregenerated.py, the script the NAMD arms are scored with,
# rather than a reimplementation, so the protocol matches by construction.
# It runs from NLST-Img because it resolves ldm/, haarPSI.py,
# configs/evaluator/vit.yaml and checkpoints/vit/42.pth relative to cwd.
#
# NAMD_3 is re-scored here too: the stored results/NAMD_3/eval_pregenerated.json
# predates --include_lpips/--include_fid/--include_haarpsi, so it has no LPIPS,
# FID or HaarPSI to compare against.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

cd /scratch/liyues_root/liyues/shxjames/NLST-Img

RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs

nvidia-smi --query-gpu=name --format=csv,noheader

for TAG in best final; do
  echo "=========== eval_pregenerated: hemexp_${TAG} ==========="
  python eval_pregenerated.py \
      --namd_dir "$RUNS/pregen_${TAG}" \
      --output_name "hemexp_${TAG}" \
      --include_lpips --include_fid --include_haarpsi
done

echo "=========== eval_pregenerated: NAMD_3 (reference, same flags) ==========="
python eval_pregenerated.py \
    --namd_dir NAMD_3 \
    --output_name NAMD_3_hemexp_ref \
    --include_lpips --include_fid --include_haarpsi

echo "=== stage 5 complete ==="
