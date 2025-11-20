#!/bin/bash -l
#SBATCH --job-name=get_thresh
#SBATCH --output=outLogs/get_human_thresholds_%j.out
#SBATCH --error=outLogs/get_human_thresholds_%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=mcdermott

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"



module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/om2/user/imgriff/projects/torch_2_aud_attn

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 -m src.get_human_thresholds

