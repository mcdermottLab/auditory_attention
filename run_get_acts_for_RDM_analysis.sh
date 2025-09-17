#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_for_RDM_%j.out
#SBATCH --error=outLogs/get_unit_acts_for_RDM_%j.err
#SBATCH --mem=100Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
##SBATCH --array=1 # 0-8 for full
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3

python3 get_acts_for_RDM_analysis.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                --model_dir /om/scratch/Thu/imgriff/acts_for_RDM_analysis \
                --n_activations 100 \
                --n_jobs 0 \
                --coch_only 
                # --time_average

