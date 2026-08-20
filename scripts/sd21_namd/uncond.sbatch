#!/bin/bash
#SBATCH --job-name=sd21-namd-uncond
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s,liyues-a100   # whichever frees first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=72:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# NAMD-on-SD2.1 stage 1: unconditional, SD2.1's own OpenCLIP-H text tower,
# NAMD aligned VAE (contrastive-0211, scale factor 0.519096).
#
# 80g is not padding: the LDM stages peaked at 50.3 GB RSS on this pipeline and
# OOM-killed at 48g. The dataset's own worker pool is most of it.
#
# Optional --export overrides: BS, LR, EPOCHS.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BS=${BS:-16}
LR=${LR:-1.0e-04}
EPOCHS=${EPOCHS:-50}
CONFIG=configs/ldm/v2.1-unconditional-unet-namd.yaml
PY=/scratch/liyues_root/liyues/shxjames/envs/NAMD/bin/python

# Size the LR schedule from the run's real step count. The scheduler takes
# absolute steps: too small a cycle_length crashes find_in_interval, too large
# and the cosine never advances and the LR stays flat at f_max. This has to run
# on the compute node -- the dataset's multiprocessing pool OOMs on a login node.
echo "== probing steps/epoch =="
STEPS=$($PY -m scripts.sd21_namd.steps_per_epoch --config $CONFIG \
          --set data.params.batch_size=${BS} trainer.max_epochs=${EPOCHS} \
        | grep '^STEPS_PER_EPOCH' | awk '{print $3}')
: "${STEPS:?steps/epoch probe failed}"
TOTAL=$(( STEPS * EPOCHS ))
CYCLE=$(( TOTAL + STEPS ))          # total + 1 epoch headroom
WARMUP=4000                         # ~1.3 epochs at bs16
echo "steps/epoch=${STEPS} total=${TOTAL} warmup=${WARMUP} cycle=${CYCLE}"

$PY run.py --config $CONFIG --train \
  --project sd21-namd-uncond-bs${BS}-lr${LR} --seed 23 --save_top_k 3 \
  --monitor_loss "val/loss" \
  --set data.params.batch_size=${BS} \
        model.base_learning_rate=${LR} \
        trainer.max_epochs=${EPOCHS} \
        model.params.scheduler_config.params.warm_up_steps=[${WARMUP}] \
        model.params.scheduler_config.params.cycle_lengths=[${CYCLE}]
