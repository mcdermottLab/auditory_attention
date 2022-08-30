#!/bin/bash
#SBATCH --job-name=get_f0s
#SBATCH --output=outLogs/get_f0_traces_%A_%a.out
#SBATCH --error=outLogs/get_f0_traces_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --array=0-22

module add openmind/miniconda

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 src/get_f0_traces.py --n_jobs 40 --array_id $SLURM_ARRAY_TASK_ID
