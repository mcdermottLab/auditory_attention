#!/bin/bash -l
#SBATCH --job-name=svm_layers
#SBATCH --output=outLogs/svm_layer_%A_%a.out
#SBATCH --error=outLogs/svm_layer_%A_%a.err
#SBATCH --array=0-8 #2-8  # Adjust based on number of layers
#SBATCH --time=1-00:00:00
#SBATCH --partition=use-everything
#SBATCH --mem=500G
#SBATCH --cpus-per-task=20


source /etc/profile.d/modules.sh
module add openmind/miniconda
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2_sva

# add src to the python path 
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the script with custom arguments
python src/run_svm_on_acts.py \
    --h5_path /om/scratch/Fri/imgriff/binaural_unit_activations_for_SVM/word_task_v10_main_feature_gain_config/word_task_v10_main_feature_gain_config_model_activations_for_word_SVM_val.h5 \
    --layer_idx $SLURM_ARRAY_TASK_ID \
    --label_key target_word_int \
    --exclude_keys layer_names \
    --k_folds 10 \
    --cv_inner 5 \
    --c_values "0.01,0.1,1" \
    --max_iter 1000 \
    --dual \
    --random_state 42 \
    --output_dir ./svm_results \
    --verbose \
    --no_dual