#!/bin/bash -l
#SBATCH --job-name=unit_anova
#SBATCH --output=outLogs/unit_anova_%A_%a.out
#SBATCH --error=outLogs/unit_anova_%A_%a.err
#SBATCH --mem=64Gb                           
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --partition=mcdermott
#SBATCH --array=2-6 # 0-6 for full

source /etc/profile.d/modules.sh
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

python3 src/unit_tuning_anova.py --model_name "word_task_v10_main_feature_gain_config_latest_ckpt" \
                --analysis_dir "/om/scratch/Thu/imgriff/binaural_unit_activations_for_anova/" \
                --layer_ix $SLURM_ARRAY_TASK_ID \
                --n_jobs $SLURM_CPUS_PER_TASK \
