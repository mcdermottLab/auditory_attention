#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=50Gb 
#SBATCH --time=3:00:00
#SBATCH --partition=ou_bcs_high
#SBATCH --cpus-per-task=20

module add miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3


export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337
