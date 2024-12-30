#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=4Gb 
#SBATCH --time=3:00:00
#SBATCH --partition=ou_bcs_low
#SBATCH --cpus-per-task=2

module add miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2
# source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch


export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337
