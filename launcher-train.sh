#!/bin/bash
#
# BSC MareNostrum5 — MAMe CNN Training Launcher
#
# Usage:
#   sbatch -A nct_367 -q acc_training launcher-train.sh
#
# To override the training config, edit the TRAIN_ARGS variable below.
#
# Workflow:
#   1. Copy the CNN/ folder to BSC:
#      scp -r /path/to/CNN/ nct01204@alogin1.bsc.es:~/DL-Lab-CNN/
#   2. On BSC, ensure data is in ~/DL-Lab-CNN/data/
#   3. Submit:
#      sbatch -A nct_367 -q acc_training launcher-train.sh
#

###  SLURM directives  ########################################################

#SBATCH --job-name="mame-train"
#SBATCH --chdir=.
#SBATCH --output=logs/mame-train_%j.out
#SBATCH --error=logs/mame-train_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1

###############################################################################

mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

# ─── Edit these for each experiment ─────────────────────────────────────────
#
# --model choices: tiny | standard_small | standard_large
#                  residual_small | residual | residual_large
#
# Phase 1 (underfitting): use standard_small or residual_small, no augment, low epochs
# Phase 2 (overfitting):  use standard_large or residual_large, no augment, more epochs
# Phase 3 (regularize):   add --augment --dropout 0.4 --weight_decay 1e-4 --scheduler cosine
#
TRAIN_ARGS="
  --data_dir    data_tiny
  --model       tiny
  --input_size  128
  --epochs      1
  --batch_size  32
  --optimizer   adam
  --lr          1e-3
  --weight_decay 0.0
  --dropout     0.0
  --num_workers 4
  --out_dir     runs/pipeline_test
"
# ─────────────────────────────────────────────────────────────────────────────

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "=============================="

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

python train.py $TRAIN_ARGS

echo "=============================="
echo "End time: $(date)"
echo "=============================="
