#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_%j.out
#SBATCH --error=outLogs/get_unit_acts_%j.err
#SBATCH --mem=8Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --partition=mcdermott
##SBATCH --array=1 # 0-8 for full
#SBATCH --gres=gpu:1 --constraint=16GB
#SBATCH -x dgx001,dgx002,node043,node091,node093

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
# python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
#                  --ckpt_path attn_cue_models/word_task_half_co_loc_v09_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=2-step=35108-v1.ckpt \
#                 --model_dir binaural_unit_tuning \
#                 --n_activations 100 \
#                 --n_jobs 0 --time_average


python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt  \
                --model_dir binaural_unit_tuning \
                --n_activations 200 \
                --n_jobs 0 --time_average

# python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_v09_cue_loc_task.yaml \
#                 --ckpt_path attn_cue_models/word_task_v09_cue_loc_task/checkpoints/epoch=0-step=6000-best_word_task-v1.ckpt \
#                 --model_dir binaural_unit_tuning \
#                 --n_activations 100 \
#                 --n_jobs 0 --time_average


# python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_v09_control_no_attn.yaml \
#                 --ckpt_path attn_cue_models/word_task_v09_control_no_attn/checkpoints/epoch=4-step=60216.ckpt \
#                 --model_dir binaural_unit_tuning \
#                 --n_activations 100 \
#                 --n_jobs 0 --time_average

# python3 get_unit_activations_for_tuning_analysis.py --config config/binaural_attn/word_task_conventional_layer_order.yaml \
#                 --ckpt_path attn_cue_models/word_task_conventional_layer_order_lr0001/checkpoints/epoch=0-step=8000-v6.ckpt \
#                 --model_dir binaural_unit_tuning \
#                 --n_activations 100 \
#                 --n_jobs 0 --time_average

