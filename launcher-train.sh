mkdir -p logs

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

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
