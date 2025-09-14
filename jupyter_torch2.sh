#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=50Gb 
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1 

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3


export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337
