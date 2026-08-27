#!/bin/bash

#SBATCH --job-name=hemexp-noflag-full
#SBATCH --output=hemexp-noflag-full.out
#SBATCH --error=hemexp-noflag-full.err
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

# Retrain the diffusion stage WITHOUT the oracle expansion flag, then run the
# complete evaluation suite on it.
#
# Why: HemExp's expansion-flag input maps to the NLST malignancy label. The
# flagged run scored ROC-AUC 0.9009 +/- 0.0093 against a 0.8136 ceiling set by
# the REAL follow-up images -- generated data cannot legitimately be more
# classifiable than what it imitates. The comparison arms (ctclip, pred_clip,
# alignvae) build their prompt from feature indices 0..14 and never see the
# label at index 15, so the flagged comparison was invalid in HemExp's favour.
#
# use_expansion_flag_input=false removes the flag embedding outright; verified
# offline that the model's output is bit-identical for flag=0 vs flag=1 while
# clinical conditioning (13 first-year biomarker + EHR dims, label already
# excluded) still changes it.
#
# The autoencoder never saw the flag, so stage 1 and its latent statistics are
# reused unchanged and only the diffusion stage is retrained.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
export WANDB_PROJECT=HEMEXP-NLST-NOFLAG

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

H=/scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/hemexp-nlst/HemExp
RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs
SEEDS="23 28 42 55 92 115 147 330 401 403 450 599 642 679 713 756 888 893 912 974"

cd "$H"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========== [1/5] retrain diffusion, no oracle flag ==========="
python train_diffusion.py CONFIGS/config_nlst_noflag.json

# train_diffusion.py stamps the run directory with a timestamp; take the newest.
D=$(ls -1dt "$RUNS"/diffusion_noflag/*__config_nlst_noflag | head -1)
echo "run dir: $D"
ls "$D"/model-*.pt | head -3

FINAL=$(ls -1v "$D"/model-[0-9]*.pt | tail -1)
echo "final checkpoint: $FINAL"

for TAG in final best; do
  if [ "$TAG" = "final" ]; then CKPT=$FINAL; else CKPT=$D/model-best.pt; fi

  echo "=========== [2/5] sample $TAG (20 seeds) ==========="
  python sample_nlst.py \
      --config CONFIGS/config_nlst_noflag.json \
      --ckpt "$CKPT" \
      --split test \
      --seeds $SEEDS \
      --out-dir "$RUNS/samples_noflag_${TAG}" \
      --export-triplets "$RUNS/eval_triplets_noflag_${TAG}"

  echo "=========== [3/5] re-frame $TAG to the common 64x64 window ==========="
  python crop_triplets.py \
      --in-dir "$RUNS/eval_triplets_noflag_${TAG}" \
      --out-dir "$RUNS/eval_triplets_noflag_${TAG}_c64"
  python to_pregen_format.py \
      --in-dir "$RUNS/eval_triplets_noflag_${TAG}_c64" \
      --out-dir "$RUNS/pregen_noflag_${TAG}"

  echo "=========== [4/5] ViT / LPIPS / FID / HaarPSI / MSE for $TAG ==========="
  cd /scratch/liyues_root/liyues/shxjames/NLST-Img
  python eval_pregenerated.py \
      --namd_dir "$RUNS/pregen_noflag_${TAG}" \
      --output_name "hemexp_noflag_${TAG}" \
      --include_lpips --include_fid --include_haarpsi

  echo "=========== [5/5] nodule_quality_eval for $TAG ==========="
  cd /scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval
  python run_evaluation.py \
      --input-dir "$RUNS/eval_triplets_noflag_${TAG}_c64" \
      --input-format npy \
      --metrics-backend pyradiomics \
      --cache-dir "cache_hemexp_noflag_${TAG}" \
      --out-dir "results_hemexp_noflag_${TAG}" \
      --out-tag "hemexp_noflag_${TAG}"

  cd "$H"
done

echo "=== no-flag retrain + full eval complete ==="
