#!/bin/bash 
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=20Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1  # --constraint=20GB
##SBATCH -x dgx001,dgx002,node093,node115

source /etc/profile.d/modules.sh
source /om2/user/rphess/miniforge3/etc/profile.d/conda.sh

export HDF5_USE_FILE_LOCKING=FALSE

conda activate /om2/user/imgriff/conda_envs/pytorch_2

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1492