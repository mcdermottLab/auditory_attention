#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=128Gb
#SBATCH --time=6:00:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=6

module add openmind/miniconda



export CONDA_ENVS_PATH=~/my-envs:/om2/user/imgriff/conda_envs

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11





export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337

