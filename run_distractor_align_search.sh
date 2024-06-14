#!/bin/bash -l 
#SBATCH --job-name=align_search
#SBATCH --output=outLogs/distractor_align_search_%j.out
#SBATCH --error=outLogs/distractor_align_search_%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1 --constraint=14GB
##SBATCH --array=0-20 # 0-20
#SBATCH -x dgx001,dgx002,node079

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

python3 gridsearch_distractor_alignment.py --config_path config/binaural_attn/word_task_half_co_loc_v07.yaml \
                --checkpoint_path attn_cue_models/word_task_half_co_loc_v07/checkpoints/epoch=2-step=46074.ckpt \
                --output_path distractor_optimization/alignment_gridsearch \
                # --opt_bg

