#!/bin/bash -l 
#SBATCH --job-name=eval_popham_swc
#SBATCH --output=outLogs/binaural_popham_swc_conds_all_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_popham_swc_conds_all_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-11 # 0-11
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
rm -r /tmp/torchinductor_imgriff


python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 \
                 --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
                 --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
                 --exp_dir popham_swc_eval_all_stim/ \

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=2-step=35108-v1.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_conventional_layer_order.yaml \
#                  --ckpt_path attn_cue_models/word_task_conventional_layer_order/checkpoints/epoch=0-step=4000-v8.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \

# python3 eval_swc_popham_2024.py --config config/binaural_attn/word_task_half_co_loc_v09_50Hz_cutoff.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_50Hz_cutoff/checkpoints/epoch=1-step=16554.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 \
#                  --stim_path /om/user/imgriff/datasets/human_swc_popham_exmpt_2024/model_eval_h5s/ \
#                  --stim_cond_map all_stim_swc_popham_exmpt_2024_cond_manifest.pkl \
#                  --exp_dir popham_swc_eval_all_stim/ \