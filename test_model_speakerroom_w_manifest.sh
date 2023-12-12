#!/bin/bash -l
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_binaural_model_80p_colocated_model_%A_%a.out
#SBATCH --error=outLogs/eval_binaural_model_80p_colocated_model_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-189 # 0-189

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 eval_binaural_w_manifest.py --config config/binaural_attn/word_task_voice_and_loc_cue_v04.yml \
                 --ckpt_path attn_cue_models/word_task_voice_and_loc_cue_v04/checkpoints/epoch=4-step=47248.ckpt \
                 --location_manifest expanded_all_azim_chosen_elev_az_11_29_2023.pkl \
                 --model_name word_task_voice_loc_cue_only_v04 --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/ \
                 --cue_type voice_and_location

