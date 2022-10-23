#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/attn_eval_timit_%A_%a.out
#SBATCH --error=outLogs/attn_eval_timit_%A_%a.err
#SBATCH --mem=24Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_timit.py --gpus 1 --n_jobs 5 --exp_dir "attn_cue_models/attn_timit_task"


