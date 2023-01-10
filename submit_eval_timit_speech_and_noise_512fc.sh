#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/attn_eval_timit_%j.out
#SBATCH --error=outLogs/attn_eval_timit_%j.err
#SBATCH --mem=60Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_timit.py --gpus 1 --n_jobs 5 --exp_dir "attn_cue_models/attn_timit_task" \
                      --model_name "match_cue_speech_and_noise_512_fc" \
                      --config_name "config/attentional_cue/attn_cue_match_target_speech_and_noise_512_fc.yaml" \
                      --ckpt_path "attn_cue_models/attn_cue_match_target_speech_and_noise_512_fc/checkpoints/epoch=0-step=20000-v2.ckpt"


