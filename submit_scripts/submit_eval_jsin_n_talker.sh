#!/bin/bash
#SBATCH --job-name=attn_cue_n_talker
#SBATCH --output=outLogs/attn_jsin_n_talkers_%A_%a.out
#SBATCH --error=outLogs/attn_jsin_n_talkers_%A_%a.err
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11

python3 eval_n_talker.py --gpus 1 --n_jobs 20 --exp_dir "attn_cue_models/attn_check_n_talkers_snr" \
                        --array_id $SLURM_ARRAY_TASK_ID \
                        --eval_cond_file "./jsin_attn_multi-distractor_conditions.pkl"
