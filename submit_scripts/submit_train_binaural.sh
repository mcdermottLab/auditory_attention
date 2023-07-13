#!/bin/bash

#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/train_binaural_attn_%j.out
#SBATCH --error=outLogs/train_binaural_attn_%j.err
#SBATCH --mem=800Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -x node[100-115]
#SBATCH --cpus-per-task=80
#SBATCH --time=11:00:00
#SBATCH --partition=multi-gpu
#SBATCH --gres=gpu:a100:8 --constraint=60GB 

# module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 train.py --config config/binaural_attn/dev_voice_and_loc_cue_001.yaml \
                 --gpus 8 --n_jobs 10  \
                 --exp_dir ./attn_cue_models/binaural_word_task_cue_voiec_and_loc_v02 \
                 --resume_training \
                 --random_seed 0
                
                
