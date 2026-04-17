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
