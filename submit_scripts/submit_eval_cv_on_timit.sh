#!/bin/bash
#SBATCH --job-name=cv_pilot
#SBATCH --output=outLogs/eval_cv_baseline_pilot_%j.out
#SBATCH --error=outLogs/eval_cv_baseline_pilot_%j.err
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_cv_pilot.py --gpus 1 --n_jobs 10 --exp_dir "attn_cue_models/cv_word_rec_pilot" \
                      --model_name "cv_clean_word_rec_baseline" \
                      --config_name config/commonvoice/cv_word_baseline.yaml \
                      --ckpt_path "attn_cue_models/cv_baseline_word_task/checkpoints/epoch=8-step=18416-v2.ckpt"\
                      --eval_timit


