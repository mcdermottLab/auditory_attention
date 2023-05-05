#!/bin/bash 
#SBATCH --job-name=cv_baseline
#SBATCH --output=outLogs/train_cv_baseline_%j.out
#SBATCH --error=outLogs/train_cv_baseline_%j.err
#SBATCH --mem=320Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:4 --constraint=60GB 

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch


python3 train.py --config config/commonvoice/cv_word_baseline.yaml\
                 --gpus 4 --n_jobs 10 --mixed_precision  \
                 --exp_dir attn_cue_models/cv_baseline_word_task 
                
                

