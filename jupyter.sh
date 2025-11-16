#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=6Gb 
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=2
#SBATCH -x node043,node084,node093,node107,node034,node109,node112,node110

source /etc/profile.d/modules.sh
module add openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch


export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337
