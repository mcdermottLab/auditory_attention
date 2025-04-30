#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_diff_sources_%A_%a.out
#SBATCH --error=outLogs/get_unit_acts_diff_sources_%A_%a.err
#SBATCH --mem=8Gb                           
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 --constraint=40GB
#SBATCH -x dgx001,dgx002,node100
#SBATCH --array=0-12 # 0-12

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff

# python3 get_acts_for_tuning_and_selection_analysis.py --config config/binaural_attn/word_task_v10_backbone_word_config.yaml \
#                 --ckpt_path attn_cue_models/word_task_v10_backbone_word_config/checkpoints/epoch=1-step=3113.ckpt \
#                 --n_activations 500 \
#                 --n_jobs 0 \
#                 --diotic \
#                 --overwrite
                # --job_id $SLURM_ARRAY_TASK_ID \
                # --overwrite \
                
python3 get_acts_for_tuning_and_selection_analysis.py --config_list model_architecture_activation_manifests/all_v10_architectures_alts_and_controls.pkl \
                --n_activations 500 \
                --n_jobs 0 \
                --diotic \
                --job_id $SLURM_ARRAY_TASK_ID \
                --overwrite \
                --cue_single_source \
                # --random_weights \
                # --time_average \

                # --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \

# python3 get_acts_for_tuning_and_selection_analysis.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
#                 --n_activations 500 \
#                 --n_jobs 0 \
#                 --diotic \
#                 --time_average \
#                 --overwrite \
                # --random_weights \
                # --cue_single_source \
                # --center_loc_only \

# python3 get_acts_for_tuning_and_selection_analysis.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
#                 --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
#                 --n_activations 100 \
#                 --n_jobs 0 \
#                 --diotic \
#                 --time_average \
                # --random_weights \



