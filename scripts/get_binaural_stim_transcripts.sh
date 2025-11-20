#!/bin/bash -l 
#SBATCH --job-name=get_tscrpt
#SBATCH --output=outLogs/get_binaural_manifest_transcripts_%j.out
#SBATCH --error=outLogs/get_binaural_manifest_transcripts_%j.err
#SBATCH --mem=8Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=mcdermott
#SBATCH --gres=gpu:a100:1 

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

##SBATCH --array=0-23 # 0-23 for 24 jobs


source /etc/profile.d/modules.sh
module load openmind8/anaconda/3-2022.10
source activate /om/user/imgriff/conda_envs/pytorch_2_whisper

which python3

python3 -m src.get_swc_binaural_manifest_transcripts --batch_size 160 \
                                                     --parent_dir /om/user/imgriff/datasets/human_word_rec_SWC_2024/ \
                                                     --manifest "full_cue_target_distractor_df_w_meta.pdpkl" \
                                                     --out_manifest_name "full_cue_target_distractor_df_w_meta_transcripts.pdpkl" \
                                                     --two_distractors



