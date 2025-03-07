#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_threshold_exmpt_v02_%A_%a.out
#SBATCH --error=outLogs/sim_human_threshold_exmpt_v02_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-59    ###58,204-206,212,255,315-317,333-345 #0-359 

#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -rf /tmp/torchinductor_imgriff

# python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=2-step=35108-v1.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_v02_only_human_locs_w_noise_mit46_1004_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_cue_noise_mit46_1004 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 
                 

# python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
#                  --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 4 --exp_dir /om/user/imgriff/projects/Auditory-Attention/binaural_eval/simulate_2024_human_threshold_experiment_v02_pink_noise_mit46_1004 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor

# python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
#                  --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor
                 

# python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_v09_cue_loc_task.yaml \
#                 --ckpt_path attn_cue_models/word_task_v09_cue_loc_task/checkpoints/epoch=0-step=2000-best_word_task-v6.ckpt \
#                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
#                 --location_idx $SLURM_ARRAY_TASK_ID \
#                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
#                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor
                 

python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor
                 

# python3 eval_sim_array_threshold_experiment_v02.py --config config/binaural_attn/word_task_half_co_loc_v09_50Hz_cutoff.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_50Hz_cutoff/checkpoints/epoch=2-step=33108.ckpt \
#                  --test_manifest binaural_test_manifests/sim_human_threshold_experiment_v02_min_reverb_room1006_30dB_bg_noise.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_threshold_experiment_v02_30_dB_pink_noise_min_verb_mit46_1004 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --pink_noise_distractor
                 
