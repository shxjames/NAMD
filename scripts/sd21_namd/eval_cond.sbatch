#!/bin/bash
#SBATCH --job-name=sd21-namd-eval
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s,liyues-a100   # whichever frees first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# eval_all.py on a NAMD-on-SD2.1 conditional checkpoint: 20-seed ViT metrics
# plus LPIPS, FID, and the per-sample (T0, T1_gen, T1_real) .npy triplets the
# nodule-property eval consumes.
#
#   sbatch --export=ALL,CKPT=checkpoints/sd21-namd-cond-bs16-lr2.0e-05/last.ckpt,TAG=last \
#          scripts/sd21_namd/eval_cond.sbatch
#
# Flags mirror worktree-sd21-baseline's scripts/sd21/eval_cond.sbatch so the
# numbers land on the same footing as the baseline's. In particular batch_size
# is left at the default 64: the 20 seeds draw noise from one RNG stream, so
# changing the batch would re-map noise onto samples and shift the metrics
# slightly even though nothing about the model changed.
#
# NOTE: eval_all.py names its output JSON by GUIDANCE ONLY, so a second run at
# the same guidance overwrites the first. Evaluate one checkpoint at a time and
# copy results/<project>/eval_all_<guidance>.json aside before running another.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${CKPT:?set CKPT to the checkpoint to evaluate}"
: "${TAG:?set TAG to a short label, e.g. last or e38}"
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }

PROJECT=${PROJECT:-sd21-namd-cond-bs16-lr2.0e-05}
CONFIG=${CONFIG:-checkpoints/${PROJECT}/config.yaml}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
PY=/scratch/liyues_root/liyues/shxjames/envs/NAMD/bin/python

echo "=========== EVAL: $CKPT (tag=$TAG) ==========="
echo "config: $CONFIG"

$PY eval_all.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --save_triplets \
    --triplet_dir "nodule_quality_eval/triplets/${PROJECT}-${TAG}" \
    --include_lpips \
    --include_fid
