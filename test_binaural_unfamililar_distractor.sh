#!/bin/bash -l 
#SBATCH --job-name=eval_unf_lang
#SBATCH --output=outLogs/binaural_unfamiliar_language_distractor_v2_stim_%A_%a.out
#SBATCH --error=outLogs/binaural_unfamiliar_language_distractor_v2_stim_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:5:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-17 # 0-17
#SBATCH -x dgx001,dgx002


module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# python3 eval_unfamiliar_distractor_stim.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --stim_manifest_path /om/user/imgriff/datasets/human_distractor_language_2024/final_stim_manifest_w_cue_tg_lang_dists.pdpkl \
#                  --test_manifest binaural_test_manifests/unfamiliar_distractor_language_1_distractor.pkl \
#                  --ckpt_pat attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir unfamiliar_distractor/ \


# python3 eval_unfamiliar_distractor_stim.py --config config/binaural_attn/word_task_v08_control_no_attn.yaml \
#                  --stim_manifest_path /om/user/imgriff/datasets/human_distractor_language_2024/final_stim_manifest_w_cue_tg_lang_dists.pdpkl \
#                  --test_manifest binaural_test_manifests/unfamiliar_distractor_language_1_distractor.pkl \
#                  --ckpt_path attn_cue_models/word_task_v08_control_no_attn/checkpoints/epoch=2-step=42504.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir unfamiliar_distractor/ \

# python3 eval_unfamiliar_distractor_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_orig.yaml \
#                  --stim_manifest_path /om/user/imgriff/datasets/human_distractor_language_2024/final_stim_manifest_w_cue_tg_lang_dists.pdpkl \
#                  --test_manifest binaural_test_manifests/unfamiliar_distractor_language_1_distractor.pkl \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_orig/checkpoints/epoch=0-step=6000-v1.ckpt \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir unfamiliar_distractor/ \


python3 eval_unfamiliar_distractor_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --stim_manifest_path /om/user/imgriff/datasets/human_distractor_language_2024/final_stim_manifest_w_cue_tg_lang_dists.pdpkl \
                 --test_manifest binaural_test_manifests/unfamiliar_distractor_language_1_distractor.pkl \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir unfamiliar_distractor/ \




# python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --ckpt_pat attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --stim_path /om/user/imgriff/datasets/human_distractor_language_2024/sounds \
#                  --array_id $SLURM_ARRAY_TASK_ID \
#                  --n_jobs 4 --exp_dir unfamiliar_distractor_short/ \

