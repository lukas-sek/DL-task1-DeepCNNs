module purge
module load miniforge
source activate deepLearning


pip install kaggle --quiet


SCRATCH_DIR="${HOME}/DL-Lab-CNN"
mkdir -p "${SCRATCH_DIR}"

python download_dataset.py --dest "${SCRATCH_DIR}/data"
