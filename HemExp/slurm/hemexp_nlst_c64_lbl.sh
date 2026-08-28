#!/bin/bash

#SBATCH --job-name=hemexp-c64-lbl
#SBATCH --output=hemexp-c64-lbl.out
#SBATCH --error=hemexp-c64-lbl.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# STRICT-PARITY ARM: retrain both stages at the benchmark's own setup.
#
# Three differences from the first arm, all of which favoured HemExp:
#   1. 64x64 CENTRE CROP, not the full 128x128 field. The benchmark arms see
#      only patch[32:96, 32:96] (verified byte-identical), so conditioning on
#      4x the context was not a fair comparison. resize_mode="crop" -- a
#      downsample does NOT reproduce their images.
#   2. NO oracle expansion flag. It maps to the malignancy label and leaked it
#      into the pixels: ROC-AUC 0.9009 against a 0.8136 real-image ceiling.
#   3. NO label-balanced batches. The NAMD arms train at the natural ~19%
#      positive rate; HemExp was training on 50/50, which is 56 steps/epoch
#      here rather than 91.
#
# Both stages retrain: the autoencoder must see the same 64x64 crops, or its
# latents are out of distribution for the diffusion stage.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

H=/scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/hemexp-nlst/HemExp
RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs
SEEDS="23 28 42 55 92 115 147 330 401 403 450 599 642 679 713 756 888 893 912 974"

cd "$H"
nvidia-smi --query-gpu=name --format=csv,noheader

AE_CKPT=$RUNS/autoencoder_c64_lbl/checkpoints/checkpoint_best.pth

# Resume guard: the first stage is expensive (~25 min) and independent of
# everything after it, so a re-run after an interruption should not repeat it --
# and must not OVERWRITE a good autoencoder with a fresh partial one. Set
# FORCE_AE=1 to retrain deliberately.
if [ -f "$AE_CKPT" ] && [ "${FORCE_AE:-0}" != "1" ]; then
  echo "=========== [1/6] SKIPPED: reusing existing autoencoder ==========="
  python - <<PY
import torch
ck = torch.load("$AE_CKPT", map_location="cpu", weights_only=False)
print(f"  step {ck['global_step']} (epoch {ck['global_step']/112:.0f}), "
      f"val recon {ck.get('val_loss'):.6f}, "
      f"label_loss_weight {ck.get('label_loss_weight')}")
PY
else
  echo "=========== [1/6] autoencoder at 64x64 crop ==========="
  python train_autoencoder_nlst.py --config CONFIGS/config_ae_nlst_c64_lbl.json
fi

test -f "$AE_CKPT" || { echo "FATAL: missing $AE_CKPT" >&2; exit 1; }

echo "=========== [2/6] latent statistics for the c64 AE ==========="
python compute_latent_stats.py \
    --ae-ckpt "$AE_CKPT" \
    --image-size 64 \
    --out "$RUNS/latent_stats_c64_lbl.json" \
    --update-config CONFIGS/config_nlst_c64_lbl.json

echo "=========== [3/6] diffusion: no flag, no balancing, 64x64 ==========="
python train_diffusion.py CONFIGS/config_nlst_c64_lbl.json

D=$(ls -1dt "$RUNS"/diffusion_c64_lbl/*__config_nlst_c64_lbl | head -1)
echo "run dir: $D"
FINAL=$(ls -1v "$D"/model-[0-9]*.pt | tail -1)
echo "final checkpoint: $FINAL"

for TAG in final best; do
  if [ "$TAG" = "final" ]; then CKPT=$FINAL; else CKPT=$D/model-best.pt; fi
  test -f "$CKPT" || { echo "skipping $TAG: $CKPT missing"; continue; }

  echo "=========== [4/6] sample $TAG (20 seeds) ==========="
  python sample_nlst.py \
      --config CONFIGS/config_nlst_c64_lbl.json \
      --ckpt "$CKPT" \
      --split test \
      --seeds $SEEDS \
      --out-dir "$RUNS/samples_c64_lbl_${TAG}" \
      --export-triplets "$RUNS/eval_triplets_c64_lbl_${TAG}"

  # Already 64x64 -- no crop_triplets step, only the layout shim.
  python to_pregen_format.py \
      --in-dir "$RUNS/eval_triplets_c64_lbl_${TAG}" \
      --out-dir "$RUNS/pregen_c64_lbl_${TAG}"

  echo "=========== [5/6] ViT / LPIPS / FID / HaarPSI / MSE for $TAG ==========="
  cd /scratch/liyues_root/liyues/shxjames/NLST-Img
  python eval_pregenerated.py \
      --namd_dir "$RUNS/pregen_c64_lbl_${TAG}" \
      --output_name "hemexp_c64_lbl_${TAG}" \
      --include_lpips --include_fid --include_haarpsi

  echo "=========== [6/6] nodule_quality_eval for $TAG ==========="
  cd /scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval
  python run_evaluation.py \
      --input-dir "$RUNS/eval_triplets_c64_lbl_${TAG}" \
      --input-format npy \
      --metrics-backend pyradiomics \
      --cache-dir "cache_hemexp_c64_lbl_${TAG}" \
      --out-dir "results_hemexp_c64_lbl_${TAG}" \
      --out-tag "hemexp_c64_lbl_${TAG}"

  cd "$H"
done

echo "=== strict-parity arm complete ==="
