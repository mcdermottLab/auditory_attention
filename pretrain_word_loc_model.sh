#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_%A_%a.err
#SBATCH --mem=100Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:2
#SBATCH --array=0-4

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source activate /om/user/imgriff/conda_envs/pytorch_2_tf

python3 pretrain.py --config config/pre_train_word_loc_task/base_word_loc_pre_train.yaml \
                 --job_id $SLURM_ARRAY_TASK_ID\
                 --gpus 2 --resume_training True \
                 --exp_dir pre_train_word_loc \
                 --set_lr_with_job 

