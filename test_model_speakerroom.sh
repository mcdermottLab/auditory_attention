#!/bin/bash
#SBATCH --job-name=eval_model
#SBATCH --output=outLogs/eval_model_%A_%a.out
#SBATCH --error=outLogs/eval_model_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1 --constraint=20GB
#SBATCH --array=0-1999

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


python3 eval_binaural.py --config /om2/user/imgriff/projects/Auditory-Attention/config/binaural_attn/word_task_mixed_cue_large_architecture_v04.yml --ckpt_path /om2/user/imgriff/projects/Auditory-Attention/attn_cue_models/word_task_mixed_cue_large_architecture_v04/checkpoints/epoch=0-step=1000-v8.ckpt \
                 --model_name word_task_mixed_cue_large --location_idx $SLURM_ARRAY_TASK_ID --cue_type both \
                 --gpus 1 --n_jobs 1 --exp_dir ./binaural_eval/large_arch_both_cue_10_26/ \

