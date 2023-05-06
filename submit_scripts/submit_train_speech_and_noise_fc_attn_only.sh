#!/bin/bash

#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/attn_cue_match_speech_and_noise_fc_attn_only_%j.out
#SBATCH --error=outLogs/attn_cue_match_speech_and_noise_fc_attn_only_%j.err
#SBATCH --mem=250Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -x node[100-115]
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1 --constraint=40GB 

export HDF5_USE_FILE_LOCKING=FALSE
module add openmind/miniconda

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python train.py --config config/attentional_cue/attn_cue_speech_and_noise_fc_only.yaml\
                 --gpus 1 --n_jobs 20 --mixed_precision  \
                 --exp_dir ./attn_cue_models/attn_cue_speech_and_noise_fc_attn_only\
                
                
