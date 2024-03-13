#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_textures
#SBATCH --output=outLogs/distractor_optimization_%A_%a.out
#SBATCH --error=outLogs/distractor_optimization_%A_%a.err
#SBATCH --mem=4Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=14GB
#SBATCH --array=1-20# 0-2
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 optimize_distractor.py --config_path config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --checkpoint_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                --output_path distractor_optimization/ \
                --n_steps 5000 --early_stop 500

