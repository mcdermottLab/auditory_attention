#!/bin/bash
#SBATCH --job-name=attn_cue_full_snr
#SBATCH --output=outLogs/attn_n_talkers_%A_%a.out
#SBATCH --error=outLogs/attn_n_talkers_%A_%a.err
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=128-159

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/torch_11_cuda_11

python3 eval_n_talker.py --gpus 1 --n_jobs 20 --exp_dir "attn_cue_models/attn_check_n_talkers_snr" --array_id $SLURM_ARRAY_TASK_ID
