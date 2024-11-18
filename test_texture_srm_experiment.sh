#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/texture_srm_expmnt_%A_%a.out
#SBATCH --error=outLogs/texture_srm_expmnt_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-839 #0-1259 for new 3 room test manifest 0-419 for min reverb 420-839 for anechoic reverb 840-1259 normal reverb
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff


# python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=2-step=35108-v1.ckpt \
#                  --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
#                  --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/texture_srm_experiment \
#                  --cue_type voice_and_location --no-overwrite --n_per_job 1 --texture_distractor --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_early_only_v09.yaml \
                 --ckpt_path attn_cue_models/word_task_early_only_v09/checkpoints/epoch=3-step=39662.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/texture_srm_experiment \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --texture_distractor --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_late_only_v09.yaml \
                 --ckpt_path attn_cue_models/word_task_late_only_v09/checkpoints/epoch=2-step=37108.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/texture_srm_experiment \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --texture_distractor --run_all_stim 

python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_v09_control_no_attn.yaml \
                 --ckpt_path attn_cue_models/word_task_v09_control_no_attn/checkpoints/epoch=0-step=10000.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_test_rooms.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/texture_srm_experiment \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1 --texture_distractor --run_all_stim 
