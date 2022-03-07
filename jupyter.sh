#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=20Gb
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott


module add openmind/miniconda/2020-01-29-py3.7



export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/imgriff/conda_envs_files

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR




export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1337

