#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mcdermott
#SBATCH --mail-user=jcruse@mit.edu
#SBATCH --mem=12Gb


module add openmind/miniconda
module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3


source activate /om2/user/jcruse/.conda/envs/jcruse_env



export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1738

