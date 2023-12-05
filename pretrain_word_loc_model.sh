#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/pretrain_word_loc_%A_%a.out
#SBATCH --error=outLogs/pretrain_word_loc_%A_%a.err
#SBATCH --mem=48Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1
##SBATCH --array=5,6#0-4

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source activate /om/user/imgriff/conda_envs/pytorch_2_tf

python3 pretrain.py --config config/pre_train_word_loc_task/base_word_loc_pre_train.yaml \
                 --gpus 1 --resume_training True \
                 --exp_dir pre_train_word_loc \

