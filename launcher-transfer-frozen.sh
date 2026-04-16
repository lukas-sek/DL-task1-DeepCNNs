#!/bin/bash
# Launcher: Transfer Learning (Frozen Feature Extractor)
# Uses ResNet18 with the backbone completely frozen. Only trains the final head.
#
# sbatch -A nct_367 -q acc_training launcher-transfer-frozen.sh

#SBATCH --job-name="transfer-frozen"
#SBATCH --chdir=.
#SBATCH --output=logs/transfer-frozen_%j.out
#SBATCH --error=logs/transfer-frozen_%j.err
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
    --epochs 15 \
    --model transfer_frozen \
    --input_size 224 \
    --batch_size 128 \
    --out_dir runs/transfer_frozen
