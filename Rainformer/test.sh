#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/%x/%x-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/logs/%x/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

export CUDA_LAUNCH_BLOCKING=1

# Set variables for testing
batch_size=64
test_ids_fn='/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/data/knmi_data/data_splits/ldcast_test_202324_avg001mm_4x20y.npy'
# MODEL_CHECKPOINT='/vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/results/model_checkpoints/rainformer_ddp_preempt/best-epoch25-loss17971.9797.ckpt'
data_path='/vol/knmimo-nobackup/restore/knmimo/thesis_pelle/data/preprocessed/rtcor_prep'
model_checkpoint=$MODEL_CHECKPOINT

source /vol/knmimo-nobackup/users/pkools/thesis-forecasting/Rainformer/.rainformer_venv/bin/activate

# Run the test script
srun python test_on_cluster.py \
    --x_seq_size=4 \
    --y_seq_size=20 \
    --batch_size=$batch_size \
    --test_IDs_fn=$test_ids_fn \
    --checkpoint_path=$MODEL_CHECKPOINT \
    --data_path=$data_path
