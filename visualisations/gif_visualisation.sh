#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --time=1:00:00
#SBATCH --mem=15G
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/visualisations/%x-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/visualisations/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

export CUDA_LAUNCH_BLOCKING=1

source ${VENV_DIR}/bin/activate

srun python gif_visualisation_outputs.py
