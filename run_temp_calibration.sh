#!/bin/bash
#SBATCH --job-name=temp_calibration
#SBATCH --output=outLogs/temp_cal%j.out
#SBATCH --error=outLogs/temp_cal%j.err
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1 --constraint=60GB
source ~/.bashrc

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python temperature_calibration.py --n_jobs 10
