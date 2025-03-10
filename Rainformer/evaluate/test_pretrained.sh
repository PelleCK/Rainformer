#!/bin/bash
#SBATCH --account=icis
#SBATCH --partition=icis
#SBATCH --qos=icis-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/%x/%x-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/%x/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

export CUDA_LAUNCH_BLOCKING=1

# Set variables for testing
batch_size=64
test_ids_fn="${DATA_SPLITS_DIR}/list_IDs_202324_avg001mm_9x9y_test_pretrained_rainformer.npy"
# MODEL_CHECKPOINT='/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/rainformer_ddp_preempt/best-epoch25-loss17971.9797.ckpt'
data_path=${PRETRAINED_DATA_DIR}
model_checkpoint=$MODEL_CHECKPOINT

source ${VENV_DIR}/bin/activate

# Run the test script
srun --export=ALL python test_pretrained.py \
    --x_seq_size=9 \
    --y_seq_size=9 \
    --window_size=9 \
    --batch_size=$batch_size \
    --test_IDs_fn=$test_ids_fn \
    --checkpoint_path=$MODEL_CHECKPOINT \
    --data_path=$data_path
