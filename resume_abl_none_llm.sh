#!/bin/bash

#SBATCH --job-name=NAMD-abl-none-r
#SBATCH --output=NAMD-abl-none-resume.out
#SBATCH --error=NAMD-abl-none-resume.err
#SBATCH --account=liyues
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

# Resume the NONE-arm run that was cancelled at epoch 4 (2026-08-27 07:35) to
# free a GPU for higher-priority work. See run_abl_none_llm.sh for the full
# rationale of the arm itself.
#
# --resume takes the project name and picks up checkpoints/<project>/last.ckpt
# (run.py:415). That path goes through trainer.fit(ckpt_path=...), which restores
# epoch, global_step, optimizer and LR-scheduler state and RNG -- NOT just the
# weights. Training continues at epoch 5 of 70 and the wandb run is continued
# rather than a new one started.
#
# The --set overrides are repeated verbatim from run_abl_none_llm.sh. They must
# match: scale_factor and ckpt_path live in the config, not the checkpoint, so a
# typo here would silently train the remaining epochs under different settings
# than the first five. scale_factor 0.395414 was verified by
# check_abl_none_scalefactor.sh (measured 0.389078 over 413,696 latent elements).
python run.py \
    --config configs/ldm/v1.5-conditional-unet-llm.yaml \
    --train \
    --project v1.5-conditional-unet-llm-abl-none \
    --resume v1.5-conditional-unet-llm-abl-none \
    --save_top_k 5 \
    --wandb \
    --set model.params.scale_factor=0.395414 \
          model.params.ckpt_path=checkpoints/unet/v1.5-unconditional-original-0211.ckpt \
          model.base_learning_rate=4e-5 \
          data.params.batch_size=16 \
          trainer.max_epochs=70
