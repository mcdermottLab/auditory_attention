#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_binaural_model_rerun_%A_%a.out
#SBATCH --error=outLogs/eval_binaural_model_rerun_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-360

module add openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 eval_binaural.py --config config/binaural_attn/dev_voice_and_loc_cue_001.yaml \
                 --ckpt_path attn_cue_models/binaural_word_task_cue_voiec_and_loc_v02/checkpoints/epoch=0-step=2000-v3.ckpt \
                 --model_name word_task_voice_loc_cue --location_idx $SLURM_ARRAY_TASK_ID \
                 --gpus 1 --n_jobs 1 --exp_dir binaural_eval/word_task_voice_loc_cue/ \

