#!/bin/bash
#SBATCH --job-name=sd21-namd-cond
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s,liyues-a100   # whichever frees first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
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
# Optional --export overrides: BS, LR, EPOCHS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${UNCOND_CKPT:?set UNCOND_CKPT to the stage-1 checkpoint}"
[ -f "$UNCOND_CKPT" ] || { echo "no such checkpoint: $UNCOND_CKPT" >&2; exit 1; }

BS=${BS:-8}
LR=${LR:-2.0e-05}
EPOCHS=${EPOCHS:-100}
CONFIG=configs/ldm/v2.1-conditional-unet-llm.yaml
PY=/scratch/liyues_root/liyues/shxjames/envs/NAMD/bin/python

# See uncond.sbatch for why this is derived rather than hardcoded.
echo "== probing steps/epoch =="
STEPS=$($PY -m scripts.sd21_namd.steps_per_epoch --config $CONFIG \
          --set data.params.batch_size=${BS} trainer.max_epochs=${EPOCHS} \
        | grep '^STEPS_PER_EPOCH' | awk '{print $3}')
: "${STEPS:?steps/epoch probe failed}"
TOTAL=$(( STEPS * EPOCHS ))
CYCLE=$(( TOTAL + STEPS ))          # total + 1 epoch headroom
WARMUP=$(( TOTAL / 20 ))            # 5% -- the concat channels and all 32
                                    # cross-attention matrices start from scratch
echo "steps/epoch=${STEPS} total=${TOTAL} warmup=${WARMUP} cycle=${CYCLE}"

$PY run.py --config $CONFIG --train \
  --project sd21-namd-cond-bs${BS}-lr${LR} --seed 23 --save_top_k 3 \
  --monitor_loss "val/loss" \
  --set model.params.ckpt_path=${UNCOND_CKPT} \
        data.params.batch_size=${BS} \
        model.base_learning_rate=${LR} \
        trainer.max_epochs=${EPOCHS} \
        model.params.scheduler_config.params.warm_up_steps=[${WARMUP}] \
        model.params.scheduler_config.params.cycle_lengths=[${CYCLE}]
