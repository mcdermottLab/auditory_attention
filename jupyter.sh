#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=8Gb 
#SBATCH --time=2:30:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH -x node104
module add openmind/miniconda


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch




export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1492

