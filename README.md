# Rainformer

Rainformer is a PyTorch-based encoder-forecaster model for precipitation nowcasting, as introduced in [Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting](https://ieeexplore.ieee.org/abstract/document/9743916). This repository is a fork of the original Rainformer repository and includes adaptations to support my thesis research on **precipitation nowcasting**, specifically focusing on high-intensity precipitation prediction. My research compares Rainformer with several state-of-the-art deep generative models, ensuring a robust evaluation under a consistent experimental framework.

---

## Key Features

- **Rainformer Model**:
  - A precipitation nowcasting model combining local and global attention mechanisms.
  - Includes channel-attention and spatial-attention modules for effective feature extraction.

- **Research Contributions**:
  - Customizations to integrate Rainformer with my specific data setup.
  - Modifications to streamline experiments comparing Rainformer with other advanced models.

---

## Requirements

This repository requires **Python 3.10.12**. All other dependencies are listed in `requirements.txt`. To set up your environment:

1. Create and activate a Python virtual environment:
   ```
   python -m venv rainformer_env
   source rainformer_env/bin/activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Data


---

## Training the Model

This repository supports both **single-GPU** and **multi-GPU** training setups. For cluster environments with **SLURM**, refer to the instructions below.

### Training on SLURM

To submit a training job on SLURM:
```
./submit_job.sh [job_name] [wandb_api_key]
```

- The `submit_job.sh` script:
  - Automatically sets up logging directories.
  - Exports the necessary environment variables for Weights & Biases tracking.
  - Submits the `train.sh` script to SLURM using `sbatch`.

- The `train.sh` script configures:
  - SLURM-specific parameters (e.g., GPUs, CPUs, memory).
  - Distributed Data Parallel (DDP) settings (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`).
  - Paths for model checkpoints and data directories.

### Training Locally on a Single GPU

For a single GPU setup without SLURM, you can run the training script directly:
```
python train_on_cluster_ddp.py \
  --x_seq_size=4 \
  --y_seq_size=20 \
  --batch_size=16 \
  --train_IDs_fn=[path_to_train_data] \
  --val_IDs_fn=[path_to_val_data] \
  --model_dir=[path_to_save_model]
```

Ensure the following for local training:
- Set `world_size=1` and `rank=0` in the configuration.
- Define `MASTER_ADDR` and `MASTER_PORT` as environment variables:
  ```
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  ```

---


## Citation

If you use Rainformer in your research, please cite the original paper:
```
@ARTICLE{9743916,
  author={Bai, Cong and Sun, Feng and Zhang, Jinglin and Song, Yi and Chen, Shengyong},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3162882}
}
```