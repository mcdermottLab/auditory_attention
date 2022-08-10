#!/bin/bash
#SBATCH --job-name=word_rec
#SBATCH --output=outLogs/cochlear_multitask_%A_%a.out
#SBATCH --error=outLogs/cochlear_multitask_%A_%a.err
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:GEFORCEGTX1080TI:2
#SBATCH --array=0-60 

module add openmind/miniconda
module add openmind/cudnn/10.1-7.6.4
module add openmind/cuda/10.2

source activate /om2/user/imgriff/conda_envs/torchaudio_11

layer_depth=$(( ($SLURM_ARRAY_TASK_ID / 10) + 5 ))
rel_date=2022-06-23
# below section needs to be modified to take
# a date as an environment var to pass into
# the script
# if [ -n "$1" ] 
# then
#    rel_date=$1
# echo ${rel_date}

python3 train.py --config model_files/${rel_date}_coch_model_${layer_depth}-layered_config_${SLURM_ARRAY_TASK_ID}.json --gpus 2 --n_jobs 10 --mixed_precision \
                 --exp_dir ./arch_search_results/coch_multitask_arch_search/${rel_date}_${SLURM_ARRAY_TASK_ID}/jsin_precombined_gammatone_40_channels_20kHz_1e-4lr_${SLURM_ARRAY_TASK_ID} \
