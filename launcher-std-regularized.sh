#!/bin/bash
# Phase 3 — Standard CNN — Regularized (best generalisation)
# Adds BatchNorm + Dropout + Augmentation + Weight decay + Cosine LR schedule
# Goal: close the gap between train and val accuracy
#
# sbatch -A nct_367 -q acc_training launcher-std-regularized.sh

#SBATCH --job-name="std-reg"
#SBATCH --chdir=.
#SBATCH --output=logs/std-reg_%j.out
#SBATCH --error=logs/std-reg_%j.err
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
  --model        standard_large \
  --input_size   128 \
  --epochs       80 \
  --batch_size   64 \
  --optimizer    adamw \
  --lr           1e-3 \
  --weight_decay 1e-4 \
  --dropout      0.4 \
  --augment \
  --scheduler    cosine \
  --num_workers  8 \
  --out_dir      runs/std_best

echo "Done | $(date)"
