#!/bin/bash
#SBATCH --job-name=multi_timit
#SBATCH --output=outLogs/attn_eval_timit_%j.out
#SBATCH --error=outLogs/attn_eval_timit_%j.err
#SBATCH --mem=60Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:1 --constraint=12GB

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_timit.py --gpus 1 --n_jobs 5 --exp_dir "attn_cue_models/attn_timit_task" \
                      --model_name "BinauralCueWLReportW_v3" \
                      --config_name /om2/user/imgriff/projects/Auditory-Attention/config/binaural_attn/word_task_mixed_cue_large_architecture.yml \
                      --ckpt_path /om2/user/imgriff/projects/Auditory-Attention/attn_cue_models/word_task_mixed_cue_large_architecture/checkpoints/epoch=0-step=2000-v2.ckpt

