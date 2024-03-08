#!/bin/bash -l
#SBATCH --job-name=get_null_corrs
#SBATCH --output=outLogs/get_fg_bg_act_%A_%a.out
#SBATCH --error=outLogs/get_fg_bg_acts_%A_%a.err
#SBATCH --mem=4Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --array=0-4 # 0-4 for full
#SBATCH -x dgx001,dgx002,node043

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 get_stage_of_attn_corrs.py --h5_path binaural_model_attn_stage_reps/word_task_half_co_loc_v07/ \
                 --job_id $SLURM_ARRAY_TASK_ID \
                 --run_each_snr 

