#!/bin/bash
# Phase 3 — Residual CNN (non-standard) — Regularized (best generalisation)
# Dropout on the GAP→FC head + augmentation + weight decay + cosine schedule
#
# sbatch -A nct_367 -q acc_training launcher-res-regularized.sh

#SBATCH --job-name="res-reg"
#SBATCH --chdir=.
#SBATCH --output=logs/res-reg_%j.out
#SBATCH --error=logs/res-reg_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1

mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Job $SLURM_JOB_ID | $(date)"

python train.py \
  --data_dir     data \
  --model        residual \
  --input_size   128 \
  --epochs       80 \
  --batch_size   64 \
  --optimizer    adamw \
  --lr           1e-3 \
  --weight_decay 1e-4 \
  --dropout      0.3 \
  --augment \
  --scheduler    cosine \
  --num_workers  8 \
  --out_dir      runs/res_best

echo "Done | $(date)"
