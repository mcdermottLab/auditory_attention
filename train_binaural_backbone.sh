#!/bin/bash
#SBATCH --job-name=train_backbone
#SBATCH --output=outLogs/train_word_task_v10_backbone_word_babble_and_noise_%j.out
#SBATCH --error=outLogs/train_word_task_v10_backbone_word_babble_and_noise_%j.err 
#SBATCH --mem=200Gb
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:4

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om/user/imgriff/conda_envs/pytorch_2_tf
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3

srun -K --cpu-bind=cores python3 spatialtrain.py --config config/binaural_attn/word_task_v10_backbone_word_babble_and_noise.yaml \
                 --gpus 4 --n_jobs 16 \
                 --exp_dir attn_cue_models \
                 --resume_training True \

