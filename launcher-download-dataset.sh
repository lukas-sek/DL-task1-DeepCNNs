#!/bin/bash
#
# BSC MareNostrum5 — Download MAMe dataset from Kaggle
#
# Usage:
#   sbatch -A nct_367 -q acc_training launcher-download-dataset.sh
#
# Before running on BSC:
#   1. Copy your ~/.kaggle/kaggle.json to BSC:
#      scp ~/.kaggle/kaggle.json nct01204@alogin1.bsc.es:~/.kaggle/kaggle.json
#   2. On BSC, set permissions:
#      chmod 600 ~/.kaggle/kaggle.json
#   3. Copy this script and download_dataset.py to your BSC working dir:
#      scp download_dataset.py launcher-download-dataset.sh nct01204@alogin1.bsc.es:~/DL-Lab-CNN/
#   4. Submit from BSC:
#      sbatch -A nct_367 -q acc_training launcher-download-dataset.sh
#
# NOTE: If BSC compute nodes have no internet access, download locally instead:
#   python download_dataset.py --dest ./data
#   scp -r ./data nct01204@alogin1.bsc.es:~/DL-Lab-CNN/data
#

###
#SBATCH --job-name="mame-download"
#SBATCH --chdir=.
#SBATCH --output=mame-download_%j.out
#SBATCH --error=mame-download_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
###

module purge
module load miniforge
source activate deepLearning

# Install kaggle package if not already present
pip install kaggle --quiet

# Download to a scratch directory (faster I/O than home)
# Adjust SCRATCH_DIR to your BSC scratch path if different
SCRATCH_DIR="${HOME}/DL-Lab-CNN"
mkdir -p "${SCRATCH_DIR}"

python download_dataset.py --dest "${SCRATCH_DIR}/data"
