#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_precedence_%A_%a.out
#SBATCH --error=outLogs/eval_precedence_%A_%a.err
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-29 # 0-29 if running 5 per job 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
#                  --test_manifest binaural_test_manifests/precedence_distractor_conditions_co_loc_conditions.pkl \

which python3
# python3 eval_precedence.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=2-step=35108-v1.ckpt \
#                  --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
#                  --model_name word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test \
#                  --cue_type voice_and_location --overwrite --n_per_job 1

# python3 eval_precedence.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt  \
#                  --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
#                  --model_name word_task_v10_main_feature_gain_config --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test \
#                  --cue_type voice_and_location --overwrite --n_per_job 1

# python3 eval_precedence.py --config config/binaural_attn/word_task_v09_control_no_attn.yaml \
#                  --ckpt_path attn_cue_models/word_task_v09_control_no_attn/checkpoints/epoch=0-step=10000.ckpt \
#                  --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
#                  --model_name word_task_v09_control_no_attn --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test \
#                  --cue_type voice_and_location --overwrite --n_per_job 1

python3 eval_precedence.py --config config/binaural_attn/word_task_half_co_loc_v09_50Hz_cutoff.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v09_50Hz_cutoff/checkpoints/epoch=2-step=33108.ckpt \
                 --test_manifest binaural_test_manifests/freymen_1999_test_conds.pkl \
                 --model_name word_task_half_co_loc_v09_50Hz_cutoff --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/precedence_distractor_test_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1

