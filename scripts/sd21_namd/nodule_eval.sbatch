#!/bin/bash
#SBATCH --job-name=sd21-namd-nodule
#SBATCH --account=liyues
#SBATCH --partition=vvh-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Nodule property eval (MedSAM2 seg + PyRadiomics) for one NAMD-on-SD2.1
# conditional checkpoint. Byte-for-byte the same pipeline and flags as
# worktree-sd21-baseline's scripts/sd21/nodule_eval.sbatch so the numbers land
# on the same footing as the SD2.1 baseline's; only ARM/TAG and the cache/out
# dirs differ. Separate cache dir per checkpoint so T1_gen masks never mix.
#
#   sbatch --export=ALL,TAG=last scripts/sd21_namd/nodule_eval.sbatch
#
# Pinned to vvh-l40s deliberately. Listing "vvh-l40s,liyues-a100" (as the other
# launchers in this dir do) made Slurm bind job 27540606 to lh1300, whose 4 GPUs
# were held for 1-2 days, and plan a start 2 days out -- even though lh1200 had
# 4 free L40S the whole time. Slurm picks one partition at submit and does not
# revisit it. `sbatch --test-only` on each partition separately shows the split.
# Walltime is 45 min against a ~9 min real runtime, to stay backfill-friendly.
#
# Consumes the triplets eval_cond.sbatch already wrote under
# nodule_quality_eval/triplets/${ARM}-${TAG}/ (20 seeds x 450 samples, each
# a (3,64,64) uint8 npy).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${TAG:?export TAG=last (or e38, e45, ...)}"
export HF_HOME=/scratch/liyues_root/liyues/shxjames/NLST-Img/cache
PY=/scratch/liyues_root/liyues/shxjames/envs/NAMD/bin/python

ARM=sd21-namd-cond-bs16-lr2.0e-05

echo "=========== NODULE EVAL: ${ARM} ${TAG} ==========="
cd nodule_quality_eval
$PY run_evaluation.py \
    --input-dir triplets/${ARM}-${TAG} \
    --input-format npy \
    --cache-dir cache_sd21_namd_${TAG} \
    --out-dir results_sd21_namd_${TAG} \
    --out-tag all

cd ..
echo "=========== METRICS: ${ARM} ${TAG} ==========="
$PY scripts/nodule_report.py nodule_quality_eval/results_sd21_namd_${TAG} all
