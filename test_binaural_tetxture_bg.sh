#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_textures
#SBATCH --output=outLogs/binaural_texture_test_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_texture_test_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-2 # 0-2
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# python3 eval_texture_backgrounds.py --config config/binaural_attn/word_task_standard_v08.yaml \
#                  --ckpt_pat attn_cue_models/word_task_standard_v08/checkpoints/epoch=0-step=6000.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir texture_mono_eval/ \

python3 eval_texture_backgrounds.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=59392.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir texture_mono_eval/ \

# python3 eval_texture_backgrounds.py --config config/binaural_attn/word_task_standard_v08.yaml \
#                  --ckpt_pat attn_cue_models/word_task_standard_v08/checkpoints/epoch=3-step=51756.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir texture_mono_eval/ \

