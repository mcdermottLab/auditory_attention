#!/bin/bash -l
#SBATCH --job-name=unit_anova
#SBATCH --output=outLogs/unit_anova_%A_%a.out
#SBATCH --error=outLogs/unit_anova_%A_%a.err
#SBATCH --mem=8Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --partition=use-everything
#SBATCH --array=0-472 # 0-472 if running 50 per job

source /etc/profile.d/modules.sh
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

python3 src/unit_variance_partitioning.py --model_name "word_task_v10_main_feature_gain_config_latest_ckpt" \
                --analysis_dir "/om/scratch/Thu/imgriff/binaural_unit_activations_for_anova/" \
                --out_dir "/om/scratch/Thu/imgriff/variance_partitioning_results/" \
                --job_array_idx $SLURM_ARRAY_TASK_ID \
                --units_per_job 50
