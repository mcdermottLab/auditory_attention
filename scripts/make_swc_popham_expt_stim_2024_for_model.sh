#!/bin/bash -l
#SBATCH --job-name=make_popham_stim
#SBATCH --output=outLogs/make_swc_popham_expmt_stim_2024_for_model_%A_%a.out
#SBATCH --error=outLogs/make_swc_popham_expmt_stim_2024_for_model_%A_%a.err
#SBATCH --mem=3Gb 
#SBATCH --time=1:30:00
#SBATCH --partition=use-everything
#SBATCH --cpus-per-task=1
#SBATCH --array=0-61

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


module load openmind8/anaconda/3-2022.10
source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

export HDF5_USE_FILE_LOCKING=FALSE

# can use default args for now 
which python3
python3 -m src.get_swc_popham_expmt_stim_2024_for_model --array_ix $SLURM_ARRAY_TASK_ID --num_manifest_rows 16
