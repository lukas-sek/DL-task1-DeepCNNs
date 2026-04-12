#!/bin/bash
# Phase 2 — Standard CNN — Overfitting
# Goal: confirm model has enough capacity to memorise training data
# Expected: train acc >> val acc  →  overfitting confirmed
# No regularisation at all: no BN, no dropout, no augmentation, no weight decay
#
# sbatch -A nct_367 -q acc_training launcher-std-overfit.sh

#SBATCH --job-name="std-overfit"
#SBATCH --chdir=.
#SBATCH --output=logs/std-overfit_%j.out
#SBATCH --error=logs/std-overfit_%j.err
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
  --model        standard_large \
  --input_size   128 \
  --epochs       60 \
  --batch_size   64 \
  --optimizer    adam \
  --lr           1e-3 \
  --weight_decay 0.0 \
  --dropout      0.0 \
  --no_batchnorm \
  --num_workers  8 \
  --out_dir      runs/std_overfit

echo "Done | $(date)"
