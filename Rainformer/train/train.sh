#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_3090:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=20G
#SBATCH --output=${GENERAL_LOG_DIR}/%x/%x-%j.out
#SBATCH --error=${GENERAL_LOG_DIR}/%x/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

config_file="./train_configs/train_config_9x9y_rtcor_all_sequences.yaml"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE: $WORLD_SIZE, MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

source ${VENV_DIR}/bin/activate

srun --kill-on-bad-exit=1 --export=ALL python train_on_cluster_ddp.py --config $config_file
