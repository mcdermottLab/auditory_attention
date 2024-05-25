#!/bin/bash -l
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:30:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1  --constraint=20GB
#SBATCH -x dgx001,dgx002

# module load /openmind/miniconda
module load openmind8/anaconda/3-2022.10


export HDF5_USE_FILE_LOCKING=FALSE

source activate /om/user/imgriff/conda_envs/pytorch_2_whisper

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1338

