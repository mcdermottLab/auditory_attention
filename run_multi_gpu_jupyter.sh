#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=high-capacity     # Any GPU on the cluster.
#SBATCH --partition=mcdermott


module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1


export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/imgriff/conda_envs_files

source activate /om2/user/imgriff/conda_envs/torchaudio_11



export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1338

