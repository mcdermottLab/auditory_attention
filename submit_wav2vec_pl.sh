#!/bin/bash
#SBATCH --job-name=wav2vec
#SBATCH --output=outLogs/wav2vec_pl_gigaspeech_%j.out
#SBATCH --error=outLogs/wav2vec_pl_gigaspeech_%j.err
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1

module add openmind/miniconda
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR

python3 train.py --config config/giga/giga_word_wav2vec_train.yaml --gpus 1 --n_jobs 32 --exp_dir ./wav2vec_1e-5_lr_exp

