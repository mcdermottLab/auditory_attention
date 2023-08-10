#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=8Gb 
#SBATCH --time=2:30:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=10
#SBATCH -x node104


source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda


export HDF5_USE_FILE_LOCKING=FALSE


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch




export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337 --NotebookApp.allow_origin='*'

