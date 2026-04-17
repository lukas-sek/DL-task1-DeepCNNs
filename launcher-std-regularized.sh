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
