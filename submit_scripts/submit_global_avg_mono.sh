#!/bin/bash

#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/attn_global_avg_%j.out
#SBATCH --error=outLogs/attn_global_avg_%j.err
#SBATCH --mem=300Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -x node[100-115]
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:2 --constraint=60GB 

module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 train.py --config config/attentional_cue/attn_cue_match_target_speech_and_noise_global_avg_cue.yaml\
                 --gpus 2 --n_jobs 10  \
                 --exp_dir ./attn_cue_models/attn_cue_match_target_speech_and_noise_global_avg_cue\
                 --resume_training
                
                
