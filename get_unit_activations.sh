#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_%j.out
#SBATCH --error=outLogs/get_unit_acts_%j.err
#SBATCH --mem=12Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=use-everything
##SBATCH --array=1 # 0-8 for full
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                --model_dir binaural_unit_tuning \
                --n_activations 100 \
                --n_jobs 0 

