#!/bin/bash
#SBATCH --job-name=make_inharm_timit
#SBATCH --output=outLogs/make_inharm_timit_%A_%a.out
#SBATCH --error=outLogs/make_inharm_timit_%A_%a.err
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --array=0-17

module add openmind/miniconda


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 src/make_inharm_timit.py --array_ix $SLURM_ARRAY_TASK_ID


