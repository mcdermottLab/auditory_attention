#!/bin/bash
#SBATCH --job-name=rms_level_check_attn
#SBATCH --output=outLogs/rms_level_check_attn_%A_%a.out
#SBATCH --error=outLogs/rms_level_check_attn_%A_%a.err
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --array=0-2

module add openmind/miniconda
module add openmind/cudnn/10.1-7.6.4
module add openmind/cuda/10.2

source activate /om2/user/imgriff/conda_envs/torchaudio_11

task_id=${SLURM_ARRAY_TASK_ID}
snr_condition=""

if [ $(($task_id % 3)) -eq 0 ]
then
    snr_condition="low"
elif [ $(($task_id % 3)) -eq 1 ]
then
    snr_condition="neutral"
else
    snr_condition="high"
fi

main_file_name="attn_cue_${snr_condition}_snr_lr_1e-4_bs_64.yaml"
exp_dir="/om2/user/jcruse/projects/End-to-end-ASR-Pytorch/attn_cue_models/attn_cue_jsin_pilot_no_pretrain_bs_64_lr_1e-4/"


python3 compare_attn_tracking.py --config config/attentional_cue/rove_rms/${main_file_name} --gpus 1 --n_jobs 20 --exp_dir $exp_dir
