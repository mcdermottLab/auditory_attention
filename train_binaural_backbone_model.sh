#!/bin/bash 
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/train_v10_backbone_word_model_%j.out
#SBATCH --error=outLogs/train_v10_backbone_word_model_%j.err # train_v08_gender_bal_4M_orig_ learned_avg
#SBATCH --mem=1000GB
#SBATCH -N 1

#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:4


# for engagning cluster
module add miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2

which python3

python3 spatialtrain.py --config config/binaural_attn/word_task_v10_backbone_word_config_w_babble.yaml \
                 --gpus 4 --n_jobs 16 --resume_training True \
                 --exp_dir attn_cue_models \

