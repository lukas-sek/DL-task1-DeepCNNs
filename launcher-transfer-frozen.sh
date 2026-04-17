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
