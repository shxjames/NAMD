#!/bin/bash

#SBATCH --job-name=NAMD-abl-none-rs
#SBATCH --output=NAMD-abl-none-resume.out
#SBATCH --error=NAMD-abl-none-resume.err
#SBATCH --account=liyues
# Same both-partitions trick as the original launcher: take whichever frees
# first. The run demonstrably fits on an L40S -- it completed 5 epochs on lh1200
# at ~5.5 min/epoch with no OOM before it was cancelled.
#SBATCH --partition=vvh-l40s,liyues-a100
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

cd /scratch/liyues_root/liyues/shxjames/NAMD-abl0

# RESUME of the NONE-arm ablation. Job 27526313 was cancelled externally at
# 2026-08-27T07:37:31, 32 min in, at epoch 4 of 70 -- healthy at the time
# (val_loss 0.322 -> 0.266 descending, no OOM, no traceback).
#
# last.ckpt was verified intact after the cancellation: 4851-entry archive,
# epoch 5, global_step 1125, with optimizer_states AND lr_schedulers present.
# run.py passes it as trainer.fit(ckpt_path=...), so this restores optimizer
# moments, LR-scheduler position, epoch counter and RNG state -- a true resume,
# not a re-init from weights. Training continues at epoch 5; ~65 epochs left,
# roughly 6 hours at the observed 5.5 min/epoch.
#
# --set values are repeated verbatim from run_abl_none_llm.sh. They must stay
# identical: scale_factor in particular is config-only and silently wrong if it
# drifts (0.395414 = the 0/0 VAE, cross-checked by job 27526314 at 0.389078).
python run.py \
    --config configs/ldm/v1.5-conditional-unet-llm.yaml \
    --train \
    --project v1.5-conditional-unet-llm-abl-none \
    --resume checkpoints/v1.5-conditional-unet-llm-abl-none/last.ckpt \
    --save_top_k 5 \
    --wandb \
    --set model.params.scale_factor=0.395414 \
          model.params.ckpt_path=checkpoints/unet/v1.5-unconditional-original-0211.ckpt \
          model.base_learning_rate=4e-5 \
          data.params.batch_size=16 \
          trainer.max_epochs=70
