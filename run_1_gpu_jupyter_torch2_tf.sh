#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=outLogs/notebook%j.out
#SBATCH --error=outLogs/notebook%j.err
#SBATCH --mem=20Gb
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH -x node055

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda


export HDF5_USE_FILE_LOCKING=FALSE

source activate /om/user/imgriff/conda_envs/pytorch_2_tf

export LC_ALL=C; unset XDG_RUNTIME_DIR && jupyter notebook --no-browser --ip='0.0.0.0' --port=1338

