#!/bin/bash
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/gain_post_norm_config_%j.out
#SBATCH --error=outLogs/gain_post_norm_config_%j.err 
#SBATCH --mem=1000Gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=mit_preemptable # ou_bcs_normal, mit_preemptable
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH -N 1 



module add miniforge

export HDF5_USE_FILE_LOCKING=FALSE

mamba activate pytorch_2
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3


srun torchrun --nproc_per_node=4  \
              spatialtrain.py --config config/binaural_attn/word_task_v10_gain_post_norm_config.yaml \
              --gpus 4 --n_jobs 8 --resume_training True \
              --exp_dir attn_cue_models \
# srun torchrun --nproc_per_node=4  \
#               spatialtrain.py --config config/binaural_attn/word_task_v10_main_feature_gain_config_half_data.yaml \
#               --gpus 4 --n_jobs 8 --resume_training True \
#               --exp_dir attn_cue_models \

