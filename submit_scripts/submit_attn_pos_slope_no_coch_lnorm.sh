#!/bin/bash
#SBATCH --job-name=attn_cue
#SBATCH --output=outLogs/attn_cue_no_ln_pos_slope_%j.out
#SBATCH --error=outLogs/attn_cue_no_ln_pos_slope_%j.err
#SBATCH --mem=400Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:A100-MCDERMOTT:2

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/torch_11_cuda_11

python3 train.py --config config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrain_slope.yml --gpus 2 --n_jobs 20 --mixed_precision \
                 --exp_dir ./attn_cue_models/attn_cue_jsin_pilot_no_pretrain_pos_slope_bs_64_lr_1e-4 \
                 
