#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=30Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1
#SBATCH -x node055
source ~/.bashrc

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch



export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1493
