#!/bin/bash 
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/train_v09_per_kernel_model_%j.out
#SBATCH --error=outLogs/train_v09_per_kernel_model_%j.err # train_v08_gender_bal_4M_orig_ learned_avg
#SBATCH --mem=200GB
#SBATCH -N 1

#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --partition=ou_bcs_high
#SBATCH --gres=gpu:8


#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

# for openmind cluster
# source /etc/profile.d/modules.sh
# module load openmind8/anaconda/3-2022.10
# source activate /om2/user/imgriff/conda_envs/pytorch_2

# for engagning cluster
module load anaconda3
source activate /orcd/data/jhm/001/imgriff/conda_envs/pytorch_2


export HDF5_USE_FILE_LOCKING=FALSE
which python3

python3 spatialtrain.py --config config/binaural_attn/word_task_v09_per_kernel_gains.yaml \
                 --gpus 8 --n_jobs 32 --resume_training True \
                 --exp_dir attn_cue_models \

