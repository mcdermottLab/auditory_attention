#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/fc_attn_only_eval_timit_%j.out
#SBATCH --error=outLogs/fc_attn_only_eval_timit_%j.err
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
                      --model_name "fc_attn_only" \
                      --config_name "config/attentional_cue/attn_cue_speech_and_noise_fc_only.yaml" \
                      --ckpt_path "attn_cue_models/attn_cue_speech_and_noise_fc_attn_only/checkpoints/epoch=0-step=36000.ckpt"