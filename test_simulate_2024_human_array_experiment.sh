#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_array_exmpt_v02_%A_%a.out
#SBATCH --error=outLogs/sim_human_array_exmpt_v02_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-419 #0-119;  0-419 for min reverb room  manifest 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=0-step=8000.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v09_cue_loc_task.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_cue_loc_task/checkpoints/epoch=0-step=6000-best_word_task-v1.ckpt \
                 --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=59392.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=59392.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_white_noise_dist \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --noise_distractor

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v08_control_no_attn.yaml \
#                  --ckpt_path attn_cue_models/word_task_v08_control_no_attn/checkpoints/epoch=2-step=42504.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_v02 \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v08_control_no_attn.yaml \
#                  --ckpt_path attn_cue_models/word_task_v08_control_no_attn/checkpoints/epoch=2-step=42504.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room_v02.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_white_noise_dist \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --noise_distractor

