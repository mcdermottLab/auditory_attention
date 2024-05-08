#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/sim_human_array_exmpt_%A_%a.out
#SBATCH --error=outLogs/sim_human_array_exmpt_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-419 #0-119;  0-419 for min reverb room
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


python3 eval_symmetric_distractors.py --config config/binaural_attn/word_task_half_co_loc_v07.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                 --test_manifest binaural_test_manifests/symmetric_distractor_conditions_w_front_back_neg_21_to_6_dBSNR_min_reverb_mit_room.pkl \
                 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/symmetric_distractor_test \
                 --cue_type voice_and_location --no-overwrite --n_per_job 1  --sim_human_array_exmpt