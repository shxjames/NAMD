#!/bin/bash

#SBATCH --job-name=NAMD-abl-none
#SBATCH --output=NAMD-abl-none.out
#SBATCH --error=NAMD-abl-none.err
#SBATCH --account=liyues
#SBATCH --partition=liyues-a100
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

# Launchers cd to $SLURM_SUBMIT_DIR by default; pin it so this always runs from
# the side checkout that carries the stripped debugger breakpoints.
cd /scratch/liyues_root/liyues/shxjames/NAMD-abl0

# NONE-arm ablation of the NAMD conditional LDM: first stage is the 0/0 VAE
# (feature_loss_weight=0.0, label_loss_weight=0.0 -- configs/autoencoder/ablation-VAE.yaml),
# carried in via the unconditional checkpoint, which bundles
# 686 model.* (UNet) + 248 first_stage_model.* (the 0/0 VAE) + 196 cond_stage_model.*.
#
# Controlled against the FULL arm v1.5-conditional-unet-NAMD-lr4e-5-bs16
# (run_namd_best.sh): every hyperparameter below is identical to that run.
# ONLY two values differ, and both are consequences of swapping the VAE:
#   scale_factor  0.519096 -> 0.395414   (latent std differs per VAE)
#   ckpt_path     ...-contrastive-0211   -> ...-original-0211
#
# The uncond UNet is in_channels=4 / context_dim=768; this stage is 8 / 2560, so
# init_from_ckpt (ddpm.py:2087) drops input_blocks.0.0.weight and the 32
# attn2.to_k/to_v tensors by shape and re-inits them. Expect ~33 dropped keys in
# the startup report -- that is by design, not a broken load.
python run.py \
    --config configs/ldm/v1.5-conditional-unet-llm.yaml \
    --train \
    --project v1.5-conditional-unet-llm-abl-none \
    --save_top_k 5 \
    --wandb \
    --set model.params.scale_factor=0.395414 \
          model.params.ckpt_path=checkpoints/unet/v1.5-unconditional-original-0211.ckpt \
          model.base_learning_rate=4e-5 \
          data.params.batch_size=16 \
          trainer.max_epochs=70
