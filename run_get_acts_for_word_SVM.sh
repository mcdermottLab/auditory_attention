#!/bin/bash -l
#SBATCH --job-name=get_acts_for_word_SVM
#SBATCH --output=outLogs/get_acts_for_word_SVM_%j.out
#SBATCH --error=outLogs/get_acts_for_word_SVM_%j.err
#SBATCH --mem=24Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=use-everything
##SBATCH --array=1 # 0-8 for full
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3

python3 get_acts_for_SVM_fits.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                --n_activations 100 \
                --n_words 50 \
                --n_jobs 0 \
                --data_split val

