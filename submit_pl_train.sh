#!/bin/bash
#SBATCH --job-name=LAS
#SBATCH --output=outLogs/LAS_pl_gigaspeech_%j.out
#SBATCH --error=outLogs/LAS_pl_gigaspeech_%j.err
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:4

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR

python3 train.py --config config/giga/giga_word_las_train_base.yaml --gpus 4 --n_jobs 24

