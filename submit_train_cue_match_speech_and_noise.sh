#!/bin/bash
#SBATCH --job-name=cue_match_speech_and_noise
#SBATCH --output=outLogs/attn_cue_match_speech_and_noise_%j.out
#SBATCH --error=outLogs/attn_cue_match_speech_and_noise_%j.err
#SBATCH --mem=300Gb
#SBATCH -N 1
##SBATCH -w dgx002
## SBATCH -w node[100-115]
#SBATCH --cpus-per-task=20
#SBATCH --time=94:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:2 --constraint=80GB 

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 train.py --config config/attentional_cue/attn_cue_match_target_speech_and_noise.yaml\
                 --gpus 2 --n_jobs 10 --mixed_precision  \
                 --exp_dir ./attn_cue_models/attn_cue_match_target_speech_and_noise 
                
