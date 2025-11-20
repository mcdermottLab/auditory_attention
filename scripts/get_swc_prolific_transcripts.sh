#!/bin/bash
#SBATCH --job-name=get_tscrpt
#SBATCH --output=outLogs/get_swc_prolific_transcripts_%A_%a.out
#SBATCH --error=outLogs/get_swc_prolific_transcripts_%A_%a.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:a100:1 
#SBATCH --array=0-40# 0-40

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"



source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10
source activate /om/user/imgriff/conda_envs/pytorch_2_whisper

python3 -m src.get_swc_prolific_manifest_transcripts --array_ix $SLURM_ARRAY_TASK_ID --batch_size 160 --overwrite 

