#!/bin/bash
#SBATCH --job-name=temp_calibration
#SBATCH --output=outLogs/run_power_analysis_%A_%a.out
#SBATCH --error=outLogs/run_power_analysis_%A_%a.err
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:30:00
#SBATCH --partition=use-everything
#SBATCH --array=0-600 #-7 # 2-7

source /etc/profile.d/modules.sh
module add openmind/miniconda

export HDF5_USE_FILE_LOCKING=FALSE
source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python src/human_loc_power_analysis.py $SLURM_ARRAY_TASK_ID
