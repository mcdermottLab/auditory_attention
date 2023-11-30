#!/bin/bash
#SBATCH --job-name=get_f0s
#SBATCH --output=outLogs/get_activations_%j.out
#SBATCH --error=outLogs/get_activations_%j.err
#SBATCH --mem=100Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1



# module add openmind/miniconda
# module add openmind/cudnn/9.1-7.0.5
# module add openmind/cuda/11.3


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

# print path to current directory

python3 get_fg_bg_acts.py --config "./config/attentional_cue/attn_cue_match_target_speech_and_noise_fc_attn_only.yaml" \
                               --model_dir "./attn_cue_models/attn_cue_match_target_speech_and_noise_fc_attn_only" \
                               --ckpt "checkpoints/epoch=0-step=36000.ckpt" \
                               --n_activations 100 \
                               --n_jobs 10 
