#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/eval_dual_channel_timit_%j.out
#SBATCH --error=outLogs/eval_dual_channel_timit_%j.err
#SBATCH --mem=60Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_timit.py --gpus 1 --n_jobs 10 --exp_dir "attn_cue_models/attn_timit_task" \
                      --model_name "dual_channel_attn" \
                      --config_name "config/control_attn/dual_channel_attn.yaml" \
                      --ckpt_path "attn_cue_models/dual_channel_attn/checkpoints/epoch=1-step=16349-v1.ckpt"