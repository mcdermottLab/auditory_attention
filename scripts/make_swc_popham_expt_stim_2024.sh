#!/bin/bash
#SBATCH --job-name=make_popham_stim
#SBATCH --output=outLogs/make_swc_popham_expmt_stim_2024_%A_%a.out
#SBATCH --error=outLogs/make_swc_popham_expmt_stim_2024_%A_%a.err
#SBATCH --mem=4Gb 
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --cpus-per-task=1
#SBATCH --array=0-143# 0-143; 144 total

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

# can use default args for now 
python -m src.get_swc_popham_expmt_stim_2024 --array_ix $SLURM_ARRAY_TASK_ID
