#!/bin/bash -l 
#SBATCH --job-name=eval_popham_swc
#SBATCH --output=outLogs/binaural_popham_swc_conds_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_swc_conds_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-11 # 0-11
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_standard_v08.yaml \
#                  --ckpt_path attn_cue_models/word_task_standard_v08/checkpoints/epoch=3-step=51756-v1.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds/ \
#                  --exp_dir popham_swc_eval/ \

# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds/ \
#                  --exp_dir popham_swc_eval/ \

# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_v08_control_no_attn.yaml \
#                  --ckpt_path attn_cue_models/word_task_v08_control_no_attn/checkpoints/epoch=2-step=42504.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds/ \
#                  --exp_dir popham_swc_eval/ \

# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_orig.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_orig/checkpoints/epoch=0-step=6000-v1.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds/ \
#                  --exp_dir popham_swc_eval/ \

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=59392.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 \
                 --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/sounds/ \
                 --exp_dir popham_swc_eval/ \

