#!/bin/bash
#SBATCH --job-name=op2wav
#SBATCH --output=outLogs/op2wav_%A_%a.out
#SBATCH --error=outLogs/op2wav_%A_%a.err
#SBATCH --mem=8000
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --array=1-100


module add openmind/ffmpeg

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

source activate /om4/group/mcdermott/user/imgriff/conda_envs_files/pytorch_ASR


python opus_to_wav_parallel.py /om2/data/public/GigaSpeech/data/opus.scp --start_ix $SLURM_ARRAY_TASK_ID --n_files 382


