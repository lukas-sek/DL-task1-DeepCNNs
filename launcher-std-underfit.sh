mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

echo "Job $SLURM_JOB_ID | $(date)"

python train.py \
  --data_dir     data \
  --model        standard_small \
  --input_size   128 \
  --epochs       30 \
  --batch_size   64 \
  --optimizer    adam \
  --lr           1e-3 \
  --weight_decay 0.0 \
  --dropout      0.0 \
  --no_batchnorm \
  --num_workers  8 \
  --out_dir      runs/std_underfit

echo "Done | $(date)"
