#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_diff_sources_%A_%a.out
#SBATCH --error=outLogs/get_unit_acts_diff_sources_%A_%a.err
#SBATCH --mem=20Gb                           
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1 
#SBATCH --array=0,4  # 0-12

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

               
python3 get_acts_for_tuning_and_selection_analysis.py --config_list model_architecture_activation_manifests/all_v10_architectures_alts_and_controls.pkl \
                --n_activations 500 \
                --n_jobs 8 \
                --diotic \
                --job_id $SLURM_ARRAY_TASK_ID \
                --cue_single_source \
                # --overwrite \
                # --random_weights \

