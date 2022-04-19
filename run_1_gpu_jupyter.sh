#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=24Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1


module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1


export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/imgriff/conda_envs_files

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/torchaudio_11



export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1338

