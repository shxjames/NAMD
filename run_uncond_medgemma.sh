#!/bin/bash

#SBATCH --job-name=NAMD-unc-mg
#SBATCH --output=NAMD-uncond-medgemma.out
#SBATCH --error=NAMD-uncond-medgemma.err
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

# NAMD stage 1: unconditional pretraining on the MedGemma empty prompt.
# Conditioning is MedGemma([BOS] + soft_prompts + [EOS]); the soft prompts train
# here and carry into stage 2 along with all 32 cross-attention K/V matrices.
#
# scale_factor (0.379953) and the PRED-VAE path are baked into the config so they
# cannot drift from stage 2. Pick LR/BS from sweep_uncond_medgemma.sh first.
#
# Env: LR, BS, EPOCHS, RESUME=1

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/NAMD

# Run from wherever sbatch was invoked, so the same script works from the main
# checkout or from a side checkout on a feature branch.
cd "${SLURM_SUBMIT_DIR:-/scratch/liyues_root/liyues/shxjames/NAMD}"

LR=${LR:-1e-4}
BS=${BS:-16}
EPOCHS=${EPOCHS:-50}
PROJ=${PROJ:-v1.5-uncond-medgemma-lr${LR}-bs${BS}}

# Full Lightning resume: weights + optimizer + scheduler + epoch counter.
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_ARGS=(--resume checkpoints/$PROJ)
fi

echo "=== $PROJ (lr=$LR bs=$BS epochs=$EPOCHS) ==="
python run.py \
    --config configs/ldm/v1.5-unconditional-unet-medgemma.yaml \
    --train \
    --project "$PROJ" \
    --save_top_k 5 \
    --wandb \
    "${RESUME_ARGS[@]}" \
    --set model.base_learning_rate=$LR \
          data.params.batch_size=$BS \
          trainer.max_epochs=$EPOCHS
