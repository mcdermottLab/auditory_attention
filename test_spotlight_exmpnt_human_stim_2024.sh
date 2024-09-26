#!/bin/bash -l 
#SBATCH --job-name=eval_binaural_swc
#SBATCH --output=outLogs/binaural_spotlight_human_stim_2024_%A_%a.out
#SBATCH --error=outLogs/binaural_spotlight_human_stim_2024_%A_%a.err
#SBATCH --mem=12Gb
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1 --constraint=20GB
#SBATCH --array=0-26 # 0-26 for standard test
#SBATCH -x dgx001,dgx002

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

# sometimes get compilation issues - remove just to be safe
rm -r /tmp/torchinductor_imgriff

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --stim_path /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/sounds \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment_human_stim/ \
                --stim_cond_map /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/human_azim_spotlight_cond_map.pkl \
                --spotlight_expmnt

rm -r /tmp/torchinductor_imgriff

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_sanity.yaml \
                 --stim_path /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/anechoic_room_sounds \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_sanity/checkpoints/epoch=7-step=89878.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment_anechoic_human_stim/ \
                 --stim_cond_map /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/human_azim_spotlight_cond_map.pkl \
                --spotlight_expmnt

rm -r /tmp/torchinductor_imgriff

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
                 --stim_path /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/sounds \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=51392.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment_human_stim/ \
                --stim_cond_map /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/human_azim_spotlight_cond_map.pkl \
                --spotlight_expmnt

rm -r /tmp/torchinductor_imgriff

python3 eval_swc_mono_stim.py --config config/binaural_attn/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout.yaml \
                 --stim_path /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/anechoic_room_sounds \
                 --ckpt_path attn_cue_models/word_task_half_co_loc_v08_gender_bal_4M_w_no_cue_learned_higher_lr_less_dropout/checkpoints/epoch=4-step=51392.ckpt \
                 --array_id $SLURM_ARRAY_TASK_ID \
                 --n_jobs 4 --exp_dir binaural_eval/simulate_2024_human_azimuth_spotlight_experiment_anechoic_human_stim/ \
                 --stim_cond_map /om/user/imgriff/datasets/human_azim_spotlight_SWC_2024/human_azim_spotlight_cond_map.pkl \
                --spotlight_expmnt

