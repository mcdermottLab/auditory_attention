#!/bin/bash -l
#SBATCH --job-name=unit_anova
#SBATCH --output=outLogs/unit_anova_%A_%a.out
#SBATCH --error=outLogs/unit_anova_%A_%a.err
#SBATCH --mem=100Gb                           
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --partition=use-everything
#SBATCH --array=38-367 # 0-92 for full

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"


source /etc/profile.d/modules.sh
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

python3 -m src.unit_tuning_anova_parallel_jsin --model_name "word_task_v10_main_feature_gain_config_latest_ckpt" \
                --analysis_dir "/om/scratch/Thu/imgriff/binaural_unit_activations_for_anova/" \
                --out_dir "/om/scratch/Fri/imgriff/binaural_unit_activations_for_anova/" \
                --job_array_idx $SLURM_ARRAY_TASK_ID \
                --n_jobs $SLURM_CPUS_PER_TASK \
                --units_per_job 64
