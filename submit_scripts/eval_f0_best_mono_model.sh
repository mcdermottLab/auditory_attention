#!/bin/bash
#SBATCH --job-name=attn_cue_f0_test
#SBATCH --output=outLogs/attn_cue_f0_test_%A_%a.out
#SBATCH --error=outLogs/attn_cue_f0_test_%A_%a.err
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-7 

module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40 
module add openmind/cuda/11.3

source activate /om2/user/imgriff/conda_envs/torch_11_cuda_11_pitch

python3 eval_f0.py --gpus 1 --n_jobs 10 --exp_dir "attn_cue_models/attn_f0_test" \
                    --array_id $SLURM_ARRAY_TASK_ID \
                    --config "config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml" \
                    --ckpt_path "attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt" \

