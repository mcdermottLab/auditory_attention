#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_binaural_model_80p_colocated_model_%A_%a.out
#SBATCH --error=outLogs/eval_binaural_model_80p_colocated_model_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=1-189 # 0-189

module add openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3



python3 eval_binaural_w_manifest.py --config config/binaural_attn/word_task_mixed_cue_v04_80p_co_located_torch_2_smaller_batch.yml \
                 --ckpt_path attn_cue_models/word_task_mixed_cue_v04_80p_co_located_torch_2_smaller_batch/checkpoints/epoch=5-step=54560.ckpt \
                 --location_manifest expanded_all_azim_chosen_elev_az_11_29_2023.pkl \
                 --model_name word_task_mixed_cue_v04_80p_co_located_torch_2_smaller_batch --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 2 --exp_dir binaural_eval/ \
                 --cue_type voice_and_location

