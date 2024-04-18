#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_binaural_speaker_room_w_manifest_%A_%a.out
#SBATCH --error=outLogs/eval_binaural_speaker_room_w_manifest_%A_%a.err
#SBATCH --mem=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-50 #0-50
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# which python3
# python3 eval_binaural_w_manifest.py --config config/binaural_attn/word_task_standard_v06sm_lr.yaml \
#                  --ckpt_path attn_cue_models/word_task_standard_v06sm_lr/checkpoints/epoch=3-step=42632.ckpt \
#                  --location_manifest binaural_test_manifests/match_human_pilot_conds.pkl \
#                  --model_name word_task_standard_v06sm_lr --location_idx $SLURM_ARRAY_TASK_ID \
#                  --gpus 1 --n_jobs 2 --exp_dir binaural_eval/human_pilot_conds \
#                  --cue_type voice_and_location --overwrite
which python3
python3 eval_binaural_w_manifest.py --config config/binaural_attn/word_task_half_co_loc_v08_70p_same.yaml \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_70p_same/checkpoints/epoch=10-step=162520.ckpt  \
                 --location_manifest /om2/user/imgriff/projects/torch_2_aud_attn/binaural_test_manifests/match_human_pilot_conds.pkl \
                 --model_name word_task_half_co_loc_v08_70p_same --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/human_pilot_conds \
                 --cue_type voice_and_location --overwrite

