#!/bin/bash
#SBATCH --job-name=train_binaural_attn
#SBATCH --output=outLogs/train_binaural_attn_v10_arch_srch_%A_%a.out
#SBATCH --error=outLogs/train_binaural_attn_v10_arch_srch_%A_%a.err
#SBATCH --mem=200GB
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4
#SBATCH --array=0-9 # 10 models in manifest


source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10

export HDF5_USE_FILE_LOCKING=FALSE

source activate /om2/user/imgriff/conda_envs/pytorch_2


# rm -r /tmp/torchinductor_imgriff

python3 spatialtrain.py --config_list /om2/user/rphess/Auditory-Attention/binaural_train_manifests/v10_50Hz_model_manifest.pkl \
                 --job_id $SLURM_ARRAY_TASK_ID \
                 --gpus 4 --n_jobs 16  --resume_training  True \
                 --exp_dir /om2/user/rphess/Auditory-Attention/attn_cue_models \
