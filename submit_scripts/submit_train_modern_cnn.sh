#!/bin/bash

#SBATCH --job-name=modern_mono
#SBATCH --output=outLogs/modern_cnn_mono_%j.out
#SBATCH --error=outLogs/modern_cnn_mono_%j.err
#SBATCH --mem=600Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=80
#SBATCH --time=12:00:00
#SBATCH --partition=multi-gpu
#SBATCH --gres=gpu:a100:8 

export HDF5_USE_FILE_LOCKING=FALSE
module add openmind/miniconda

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python train.py --config config/attentional_cue/attn_modern_cnn_speech_and_noise.yaml \
                 --gpus 8 --n_jobs 10  \
                 --exp_dir ./attn_cue_models/attn_cue_speech_and_noise_modern_cnn \
                
                
