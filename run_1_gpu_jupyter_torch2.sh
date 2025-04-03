#!/bin/bash -l
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --partition=ou_bcs_high
#SBATCH --gres=gpu:h100:1

module load miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1338

