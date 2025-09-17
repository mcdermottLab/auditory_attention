#!/bin/bash
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/gain_post_norm_config_%j.out
#SBATCH --error=outLogs/gain_post_norm_config_%j.err 
#SBATCH --mem=100Gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=mcdermott # ou_bcs_normal, mit_preemptable
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1 



source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2



# srun torchrun --nproc_per_node=4  \
#               spatialtrain.py --config config/binaural_attn/word_task_v10_gain_post_norm_config.yaml \
#               --gpus 4 --n_jobs 8 --resume_training True \
#               --exp_dir attn_cue_models \
srun torchrun --nproc_per_node=4  \
              spatialtrain.py --config config/binaural_attn/word_task_v10_main_feature_gain_config_half_data.yaml \
              --gpus 4 --n_jobs 8 --resume_training True \
              --exp_dir attn_cue_models \

