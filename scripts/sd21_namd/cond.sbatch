#!/bin/bash
#SBATCH --job-name=sd21-namd-cond
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s,liyues-a100   # whichever frees first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120g   # bs16 peaked at 97 GiB against a 100 G request
#SBATCH --time=72:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# NAMD-on-SD2.1 stage 2: conditional, MedGemma cross-attention, seeded from the
# stage-1 checkpoint.
#
# Per the select-late rule, seed from stage 1's last.ckpt:
#   sbatch --export=ALL,UNCOND_CKPT=checkpoints/sd21-namd-uncond-bs16-lr1.0e-04/last.ckpt \
#          scripts/sd21_namd/cond.sbatch
#
# Optional --export overrides: BS, LR, EPOCHS, PROJECT.
#
# A 1-epoch smoke before committing 13 h, into its own project so it cannot
# write into the real run's checkpoint dir:
#   sbatch --export=ALL,UNCOND_CKPT=...,EPOCHS=1,PROJECT=sd21-namd-cond-smoke \
#          scripts/sd21_namd/cond.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${UNCOND_CKPT:?set UNCOND_CKPT to the stage-1 checkpoint}"
[ -f "$UNCOND_CKPT" ] || { echo "no such checkpoint: $UNCOND_CKPT" >&2; exit 1; }

BS=${BS:-16}
LR=${LR:-2.0e-05}
EPOCHS=${EPOCHS:-100}
PROJECT=${PROJECT:-sd21-namd-cond-bs${BS}-lr${LR}}
CONFIG=configs/ldm/v2.1-conditional-unet-llm.yaml
PY=/scratch/liyues_root/liyues/shxjames/envs/NAMD/bin/python

# No steps/epoch probe here, unlike uncond.sbatch: this stage runs a constant LR,
# so nothing consumes the step count. That also retires the probe's failure mode
# (it needs a retriever for the conditional split) and the ~3 min it cost.

$PY run.py --config $CONFIG --train \
  --project ${PROJECT} --seed 23 --save_top_k 3 \
  --monitor_loss "val/loss" \
  --set model.params.ckpt_path=${UNCOND_CKPT} \
        data.params.batch_size=${BS} \
        model.base_learning_rate=${LR} \
        trainer.max_epochs=${EPOCHS}
