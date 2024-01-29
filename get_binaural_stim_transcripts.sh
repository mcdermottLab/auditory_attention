#!/bin/bash -l 
#SBATCH --job-name=get_tscrpt
#SBATCH --output=outLogs/get_binaural_manifest_transcripts_%j.out
#SBATCH --error=outLogs/get_binaural_manifest_transcripts_%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 
##SBATCH --array=35-39 # 0-40


source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10
source activate /om/user/imgriff/conda_envs/pytorch_2_whisper

which python3
python3 src/get_swc_binaural_manifest_transcripts.py --batch_size 160

