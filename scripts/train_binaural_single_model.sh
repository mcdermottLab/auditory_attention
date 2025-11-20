#!/bin/bash
#SBATCH --job-name=train_new_binaural_attn
#SBATCH --output=outLogs/train_v11_main_feature_gain_arch_%j.out
#SBATCH --error=outLogs/train_v11_main_feature_gain_arch_%j.err # train_v09_gender_bal_4M_w_no_cue_learned_ word_task_v09_cue_loc_task_
#SBATCH --mem=100Gb
#SBATCH -N 1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"



#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=mcdermott   # multi-gpu
#SBATCH --gres=gpu:a100-mcdermott:4

##SBATCH --mem=100Gb
##SBATCH --cpus-per-task=16
##SBATCH --partition=normal
##SBATCH --time=2-00:00:00
##SBATCH --gres=gpu:a100:4

#source /etc/profile.d/modules.sh
#module use /cm/shared/modulefiles

source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2
#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3


python3 -m src.spatialtrain --config config/binaural_attn/word_task_v10_main_feature_gain_config.yaml \
                 --gpus 4 --n_jobs 4 --resume_training True \
                 --exp_dir attn_cue_models \

