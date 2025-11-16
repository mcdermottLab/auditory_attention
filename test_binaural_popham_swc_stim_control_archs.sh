#!/bin/bash -l 
#SBATCH --job-name=eval_popham_swc
#SBATCH --output=outLogs/binaural_popham_swc_conds_all_stim_control_arch_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_swc_conds_all_stim_control_arch_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-11 # 0-11
#SBATCH -x dgx001,dgx002,node104

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
rm -r /tmp/torchinductor_imgriff


python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_half_co_loc_v09_50Hz_cutoff.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v09_50Hz_cutoff/checkpoints/epoch=2-step=33108.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 \
                 --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
                 --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
                 --exp_dir popham_swc_eval_all_stim/ \
                #  --overwrite

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_early_only_v10.yaml \
#                  --ckpt_path attn_cue_models/word_task_early_only_v10/checkpoints/epoch=7-step=92753.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \
#                 #  --overwrite

# rm -r /tmp/torchinductor_imgriff

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_late_only_v10.yaml \
#                  --ckpt_path attn_cue_models/word_task_late_only_v10/checkpoints/epoch=7-step=96753.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \
#                 #  --overwrite


# rm -r /tmp/torchinductor_imgriff

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_v10_control_no_attn.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_control_no_attn/checkpoints/epoch=7-step=94753.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \
                #  --overwrite


