#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=6Gb
#SBATCH -n 1 
#SBATCH --time=6:00:00
#SBATCH --partition=mcdermott
#SBATCH --cpus-per-task=1
#SBATCH -x node[100-115]
module add openmind/miniconda



export CONDA_ENVS_PATH=~/my-envs:/om2/user/imgriff/conda_envs

source activate /om2/user/imgriff/conda_envs/aligner




export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337

