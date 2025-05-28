#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=12Gb 
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH -x dgx001,dgx002

source /etc/profile.d/modules.sh
module add openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva


export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337
