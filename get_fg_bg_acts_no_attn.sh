#!/bin/bash -l
#SBATCH --job-name=get_fg_bg_acts
#SBATCH --output=outLogs/get_fg_bg_act_%j.out
#SBATCH --error=outLogs/get_fg_bg_acts_%j.err
#SBATCH --mem=60Gb                           # Logs say job uses 30Gb, get OOM if less than 48Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=12GB
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 get_fg_bg_acts_v2.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                 --model_dir binaural_model_attn_stage_reps \
                 --n_jobs 0 \
                 --n_activations 100 \
                 --no-attention

