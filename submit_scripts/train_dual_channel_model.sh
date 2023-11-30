#!/bin/bash

#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/train_dual_channel_model_%j.out
#SBATCH --error=outLogs/train_dual_channel_model_%j.err
#SBATCH --mem=250Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -x node[100-115]
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:4 --constraint=60GB 

export HDF5_USE_FILE_LOCKING=FALSE
module add openmind/miniconda

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python train.py --config config/control_attn/dual_channel_attn.yaml\
                 --gpus 4 --n_jobs 10 --mixed_precision  \
                 --exp_dir attn_cue_models/dual_channel_attn\
                
