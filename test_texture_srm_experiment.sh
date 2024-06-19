#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/texture_srm_expmnt_%A_%a.out
#SBATCH --error=outLogs/texture_srm_expmnt_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-1259 #0-1259 for new 3 room test manifest 
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff



python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/texture_srm_experiment \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --texture_distractor --run_all_stim 

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment_white_noise_dist \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim --noise_distractor
#                 #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \

# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v08.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v08/checkpoints/epoch=2-step=34504-v1.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_in_elev_neg_21_to_6_dBSNR_min_reverb_room.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/simulate_2024_human_array_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --sim_human_array_exmpt --run_all_stim
#                 #  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \