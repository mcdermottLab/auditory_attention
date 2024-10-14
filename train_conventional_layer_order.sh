#!/bin/bash -l  
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/word_task_conventional_layer_order_w_onecycle_sch%j.out
#SBATCH --error=outLogs/word_task_conventional_layer_order_w_onecycle_sch%j.err # train_v08_gender_bal_4M_orig_ learned_avg
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100-mcdermott:4


#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

# for openmind cluster
source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10
source activate /om2/user/imgriff/conda_envs/pytorch_2


export HDF5_USE_FILE_LOCKING=FALSE
which python3

# python3 spatialtrain.py --config config/binaural_attn/word_task_conventional_layer_order.yaml \
#                  --gpus 4 --n_jobs 16 \
#                  --exp_dir attn_cue_models \
#                 #   --resume_training True \

# python3 spatialtrain.py --config config/binaural_attn/word_task_conventional_layer_order_lr00005.yaml \
#                  --gpus 8 --n_jobs 32 \
#                  --exp_dir attn_cue_models \
#                   --resume_training True \
                  
# python3 spatialtrain.py --config config/binaural_attn/word_task_conventional_layer_order_lr001.yaml \
#                  --gpus 4 --n_jobs 16 \
#                  --exp_dir attn_cue_models \
#                   --resume_training True \
                  
python3 spatialtrain.py --config config/binaural_attn/word_task_conventional_layer_order_w_onecycle_sch.yaml \
                 --gpus 4 --n_jobs 16 \
                 --exp_dir attn_cue_models \
                  --resume_training True \


