#!/bin/bash

#SBATCH --job-name=hemexp-noflag
#SBATCH --output=hemexp-noflag.out
#SBATCH --error=hemexp-noflag.err
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=60GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=BEGIN,END

# CONTROL: regenerate with the oracle expansion flag held CONSTANT.
#
# The default sampling passes the dataset's true flag, which for NLST IS the
# malignancy label. The ViT then scored the generated images at ROC-AUC ~0.90,
# above the 0.8136 ceiling set by the REAL follow-up images -- generated data
# cannot legitimately be more classifiable than what it imitates, so the label
# was leaking through the flag embedding into the pixels.
#
# The arms this is compared against (ctclip, pred_clip, alignvae) condition on
# a text prompt built from feature indices 0..14, which EXCLUDES the label at
# index 15. So they never see it and HemExp did, making the AUROC comparison
# invalid in HemExp's favour.
#
# --flag-override 0 fixes the flag for every subject, so it carries no
# label information and the AUROC becomes an honest measure of what the image
# content alone supports. This is also the regime HemExp's own README describes
# for inference when expansion status is unknown.

set -euo pipefail

export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache

source ~/anaconda3/etc/profile.d/conda.sh
module load cuda
conda activate /scratch/liyues_root/liyues/shxjames/envs/HEMEXP

H=/scratch/liyues_root/liyues/shxjames/NAMD/.claude/worktrees/hemexp-nlst/HemExp
RUNS=/scratch/liyues_root/liyues/shxjames/NAMD/HemExp_NLST_runs
D=$RUNS/diffusion/20260825_213513__config_nlst
SEEDS="23 28 42 55 92 115 147 330 401 403 450 599 642 679 713 756 888 893 912 974"

cd "$H"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========== sampling final with flag held at 0 ==========="
python sample_nlst.py \
    --config CONFIGS/config_nlst.json \
    --ckpt "$D/model-30.pt" \
    --split test \
    --seeds $SEEDS \
    --flag-override 0 \
    --out-dir "$RUNS/samples_final_noflag" \
    --export-triplets "$RUNS/eval_triplets_final_noflag"

echo "=========== re-frame to the common 64x64 window ==========="
python crop_triplets.py \
    --in-dir "$RUNS/eval_triplets_final_noflag" \
    --out-dir "$RUNS/eval_triplets_final_noflag_c64"

python to_pregen_format.py \
    --in-dir "$RUNS/eval_triplets_final_noflag_c64" \
    --out-dir "$RUNS/pregen_final_noflag"

echo "=========== ViT / LPIPS / FID / HaarPSI / MSE ==========="
cd /scratch/liyues_root/liyues/shxjames/NLST-Img
python eval_pregenerated.py \
    --namd_dir "$RUNS/pregen_final_noflag" \
    --output_name hemexp_final_noflag \
    --include_lpips --include_fid --include_haarpsi

echo "=========== nodule_quality_eval ==========="
cd /scratch/liyues_root/liyues/shxjames/NAMD/nodule_quality_eval
python run_evaluation.py \
    --input-dir "$RUNS/eval_triplets_final_noflag_c64" \
    --input-format npy \
    --metrics-backend pyradiomics \
    --cache-dir cache_hemexp_final_noflag \
    --out-dir results_hemexp_final_noflag \
    --out-tag hemexp_final_noflag

echo "=== control complete ==="
