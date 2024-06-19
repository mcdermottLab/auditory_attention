#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_array_exmpt_%A_%a.out
#SBATCH --error=outLogs/sim_human_array_exmpt_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-1259 #0-1259 for new 3 room test manifest 
##SBATCH --array=0-419 #0-119;  0-419 for min reverb room  manifest 
##SBATCH --array=0-89 #0-89 min reverb room symmetric in elevation  
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --test_manifest binaural_test_manifests/sim_2024_human_experiment.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt


# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_min_reverb_mit_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_orig.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_orig/checkpoints/epoch=0-step=4000-v1.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_in_elev_neg_21_to_6_dBSNR_min_reverb_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_sanity_check \
#                  --cue_type voice_and_location --overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_sanity_check \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim
#                 #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
#                  --test_manifest binaural_test_manifests/human_array_exmpt_sim_w_front_back_21_to_6_dBSNR_min_reverb_mit_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_w_noise_diff_stim \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --modulated_ssn_distractors
#                 #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms_all_symmetric.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_sanity_check \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms_all_symmetric.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_white_noise_dist \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --noise_distractor
                #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08/checkpoints/epoch=2-step=34504-v1.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_in_elev_neg_21_to_6_dBSNR_min_reverb_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim
#                 #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \