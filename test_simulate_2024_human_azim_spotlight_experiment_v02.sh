#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_spotlight_expmt_v02_%A_%a.out
#SBATCH --error=outLogs/sim_spotlight_expmt_v02_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=3-14 #0-15 for all azimuth conditions 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

# python3 eval_sim_array_spotlight_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679.ckpt \
#                  --test_manifest binaural_test_manifests/sim_azim_spotlight_v02_normal_room1004.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/sim_azim_spotlight_v02_normal_room1004 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --run_all_stim --no-pink_noise_bg

# python3 eval_sim_array_spotlight_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
#                  --test_manifest binaural_test_manifests/sim_azim_spotlight_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/sim_azim_spotlight_v02_min_reverb_room1004_no_bg \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --run_all_stim --no-pink_noise_bg

# python3 eval_sim_array_spotlight_experiment_v02_use_aud_transforms.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                  --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679.ckpt \
#                  --test_manifest binaural_test_manifests/sim_azim_spotlight_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/sim_azim_spotlight_v02_min_reverb_room1004_no_bg_aud_transforms \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --run_all_stim --no-pink_noise_bg

python3 eval_sim_array_spotlight_experiment_v02.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                 --test_manifest binaural_test_manifests/sim_azim_spotlight_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/sim_azim_spotlight_v02_min_reverb_room1004_30dB_pink_noise_bg \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --run_all_stim --pink_noise_bg
