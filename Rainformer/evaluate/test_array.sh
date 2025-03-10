#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --output=${GENERAL_LOG_DIR}/%x/%x-%A_%a.out
#SBATCH --error=${GENERAL_LOG_DIR}/%x/%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

export CUDA_LAUNCH_BLOCKING=1

# lead times to evaluate on
LEAD_TIMES=(1 2 4 6 9)

# Get lead time based on the array task ID
LEAD_TIME="${LEAD_TIMES[SLURM_ARRAY_TASK_ID]}"

CONFIG_FILE="./evaluation_configs/evaluation_config_6x9y_rtcor_from_9x9y_heavy.yaml"

source ${VENV_DIR}/bin/activate

# Run the test script with the given config file and lead time
srun --export=ALL python test_on_cluster.py --config_path=$CONFIG_FILE --lead_time=$LEAD_TIME
