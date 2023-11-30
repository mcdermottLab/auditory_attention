#!/bin/bash

#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/train_mono_cv_attn_%j.out
#SBATCH --error=outLogs/train_mono_cv_attn_%j.err
#SBATCH --mem=400Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -x node[100-115]
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4 

# module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 train.py --config config/binaural_attn/dev_voice_and_loc_cue_001_mono_cntrl.yaml \
                 --gpus 4 --n_jobs 10  \
                 --exp_dir ./attn_cue_models/mono_attn_cv_scenes \
                 --random_seed 0 --resume_training
                
                
