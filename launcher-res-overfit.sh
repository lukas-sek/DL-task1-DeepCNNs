#!/bin/bash
# Phase 2 — Residual CNN (non-standard) — Overfitting
# Goal: confirm large residual network memorises training data
# Expected: train acc >> val acc
# ResidualCNN always uses BN (built into residual blocks) — dropout set to 0 instead
#
# sbatch -A nct_367 -q acc_training launcher-res-overfit.sh

#SBATCH --job-name="res-overfit"
#SBATCH --chdir=.
#SBATCH --output=logs/res-overfit_%j.out
#SBATCH --error=logs/res-overfit_%j.err
#SBATCH --time=04:00:00
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
  --model        residual_large \
  --input_size   128 \
  --epochs       60 \
  --batch_size   64 \
  --optimizer    adam \
  --lr           1e-3 \
  --weight_decay 0.0 \
  --dropout      0.0 \
  --num_workers  8 \
  --out_dir      runs/res_overfit

echo "Done | $(date)"
