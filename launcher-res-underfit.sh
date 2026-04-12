#!/bin/bash
# Phase 1 — Residual CNN (non-standard) — Underfitting
# Goal: confirm a small residual network underfits MAMe
# Expected: low train acc, val acc ≈ train acc
#
# sbatch -A nct_367 -q acc_training launcher-res-underfit.sh

#SBATCH --job-name="res-underfit"
#SBATCH --chdir=.
#SBATCH --output=logs/res-underfit_%j.out
#SBATCH --error=logs/res-underfit_%j.err
#SBATCH --time=02:00:00
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
  --model        residual_small \
  --input_size   128 \
  --epochs       30 \
  --batch_size   64 \
  --optimizer    adam \
  --lr           1e-3 \
  --weight_decay 0.0 \
  --dropout      0.0 \
  --num_workers  8 \
  --out_dir      runs/res_underfit

echo "Done | $(date)"
