#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook_%j.out
#SBATCH --error=outLogs/notebook_%j.err
#SBATCH --mem=20Gb 
#SBATCH --time=6:00:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=10
#SBATCH -x node104


source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch




export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1492
