#!/bin/bash
# Launcher: Transfer Learning (Full Fine-tuning)
# Uses ResNet18 and finetunes all layers.
# Note the lower learning rate parameter so pre-trained weights aren't immediately destroyed.
#
# sbatch -A nct_367 -q acc_training launcher-transfer-finetune.sh

#SBATCH --job-name="transfer-finetune"
#SBATCH --chdir=.
#SBATCH --output=logs/transfer-finetune_%j.out
#SBATCH --error=logs/transfer-finetune_%j.err
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
    --data_dir data \
    --epochs 30 \
    --model transfer_finetune \
    --input_size 224 \
    --batch_size 128 \
    --lr 1e-4 \
    --out_dir runs/transfer_finetune
