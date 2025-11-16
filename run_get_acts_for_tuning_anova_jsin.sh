#!/bin/bash -l
#SBATCH --job-name=get_unit_acts
#SBATCH --output=outLogs/get_unit_acts_for_anova_%j.out
#SBATCH --error=outLogs/get_unit_acts_for_anova_%j.err
#SBATCH --mem=20Gb                           
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1 
##SBATCH --array=12  # 0-12

module load openmind8/anaconda/3-2022.10
export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2

which python3
rm -r /tmp/torchinductor_imgriff


python3 get_acts_for_tuning_anova_jsin.py --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                --ckpt_path attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt \
                --n_activations 5000 \
                --n_jobs 8 \
                --time_average \
                --overwrite



