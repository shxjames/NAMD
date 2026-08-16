#!/bin/bash

#SBATCH --job-name=NAMD-unc-mg-sweep
#SBATCH --output=NAMD-uncond-mg-sweep-%a.out
#SBATCH --error=NAMD-uncond-mg-sweep-%a.err
#SBATCH --account=liyues
#SBATCH --partition=liyues-a100
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3%2

# NAMD stage 1 LR x BS screen: 2 lr x 2 bs, 15 epochs each, top-1 ckpt.
# Centred on the SD1.5 unconditional template default (lr 1e-4, bs 16).
# Rank by min val/loss, then run the winner to 50 epochs with
# run_uncond_medgemma.sh.

LRS=(5e-5 1e-4)
BSS=(16 32)
LR=${LRS[$((SLURM_ARRAY_TASK_ID / 2))]}
BS=${BSS[$((SLURM_ARRAY_TASK_ID % 2))]}
PROJ=uncond-mg-screen-lr${LR}-bs${BS}

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

# Run from wherever sbatch was invoked, so the same script works from the main
# checkout or from a side checkout on a feature branch.
cd "${SLURM_SUBMIT_DIR:-/scratch/liyues_root/liyues/shxjames/NAMD}"

echo "=== $PROJ ==="
python run.py \
    --config configs/ldm/v1.5-unconditional-unet-medgemma.yaml \
    --train \
    --project $PROJ \
    --save_top_k 1 \
    --wandb --offline \
    --set model.base_learning_rate=$LR \
          data.params.batch_size=$BS \
          trainer.max_epochs=15
