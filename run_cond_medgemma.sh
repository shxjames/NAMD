#!/bin/bash

#SBATCH --job-name=NAMD-cond-mg
#SBATCH --output=NAMD-cond-medgemma.out
#SBATCH --error=NAMD-cond-medgemma.err
#SBATCH --account=liyues
#SBATCH --partition=liyues-a100
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# NAMD stage 2: conditional, initialised exactly from the stage-1 MedGemma
# unconditional checkpoint. lr 2e-5 / bs 8 (the established recipe), 65 epochs.
#
# strict_load: true means the run dies at startup rather than silently dropping
# weights, so a config mismatch costs seconds instead of a day. Expect one line:
#   Expanded ...input_blocks.0.0.weight (320, 4, 3, 3) -> (320, 8, 3, 3)
# followed by "exact load, N tensors, 0 missing, 0 unexpected, 0 dropped".
#
# STAGE1 defaults to last.ckpt rather than the best-val checkpoint: for the UNet
# stages val/loss is a poor selector and late checkpoints do better.
#
# Env: STAGE1 (path to the stage-1 checkpoint), PROJ, RESUME=1

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

# Run from wherever sbatch was invoked, so the same script works from the main
# checkout or from a side checkout on a feature branch.
cd "${SLURM_SUBMIT_DIR:-/scratch/liyues_root/liyues/shxjames/NAMD}"

STAGE1="${STAGE1:?set STAGE1 to the stage-1 checkpoint, e.g. checkpoints/v1.5-uncond-medgemma-lr1e-4-bs16/last.ckpt}"
PROJ=${PROJ:-v1.5-cond-medgemma}

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_ARGS=(--resume checkpoints/$PROJ)
fi

echo "=== $PROJ (stage 1: $STAGE1) ==="
python run.py \
    --config configs/ldm/v1.5-conditional-unet-llm-strict.yaml \
    --train \
    --project "$PROJ" \
    --save_top_k 5 \
    --wandb \
    "${RESUME_ARGS[@]}" \
    --set model.params.ckpt_path="$STAGE1"
