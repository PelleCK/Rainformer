import os
import sys

import inspect
import random
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from fire import Fire
from omegaconf import OmegaConf
from tqdm import tqdm

# local imports
project_dir = os.getenv('PROJECT_DIR')
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'Rainformer'))
import dataloading as dl
from Rainformer import Net
from tool import *

# from tqdm import tqdm, trange

#TODO: ensure that logging also continues from previous run
#TODO: run deterministic experiments

def set_seeds(seed, reproducible=False):
    """ Set random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = reproducible 
    torch.backends.cudnn.benchmark = not reproducible
    if dist.is_initialized():
        torch.cuda.set_device(dist.get_rank())


def setup_distributed(rank: int, world_size: int) -> None:
    """ Initialize distributed training

    MASTER_ADDR and MASTER_PORT are set as environment variables in the SLURM script.

    Args:
    
        rank: Rank of the current process
        world_size: Total number of processes (gpus) used for training
    """
    if world_size > 1: # only if on multi-gpu setup
        master_addr = os.getenv('MASTER_ADDR')
        master_port = os.getenv('MASTER_PORT')

        dist.init_process_group(
            "nccl", 
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank, 
            world_size=world_size)
        
        torch.cuda.set_device(rank)
        
        print(f"Process group initialized with rank {rank} and world_size {world_size}", flush=True)

def cleanup():
    """ Cleanup distributed training """
    if dist.is_initialized():
        print("Destroying process group", flush=True)
        dist.destroy_process_group()


def log_missing_hyperparams(func, run, ignore=[]):
    # Get the default argument values from the function signature
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if param.name not in run.config and param.name not in ignore:
            # Add the missing arguments with their default values to wandb.config
            run.config[param.name] = param.default


def check_ddp_model_synchronization(net, rank):
    """ Check if model parameters are synchronized across ranks """

    # Iterate over model parameters
    for param in net.parameters():
        # Clone the parameter tensor to avoid modifying the original tensor
        param_tensor = param.detach().clone()

        # Reduce the parameter tensors across all processes (sum them)
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)

        # Divide by world size to get the average across all GPUs
        param_tensor /= dist.get_world_size()

        # Check if the current rank's parameter is equal to the average
        if not torch.allclose(param_tensor, param.detach(), atol=1e-6):
            print(f"Rank {rank}: Model parameters are not synchronized.")
            return False
    
    if rank == 0:
        print(f"Rank {rank}: Model parameters are synchronized.")
    return True

def adjust_learning_rate(lr_orig, batch_size_orig, base_batch_size, world_size, y_seq_size_orig, y_seq_size):
    seq_increase = y_seq_size / y_seq_size_orig
    effective_batch_size = base_batch_size * world_size
    batch_increase = effective_batch_size / batch_size_orig
    lr = lr_orig / (seq_increase * batch_increase)
    return lr

# def log_hyperparams(config, optimizer, scheduler, rank):
#     wandb.init(
#         project='thesis-forecasting',
#         name=os.getenv('SLURM_JOB_NAME', 'ddp-run'),
#         config=config
#     )


# -----------------------------------------------
def setup_data_module(
    train_IDs_fn: str,
    val_IDs_fn: str,
    test_IDs_fn: str,
    data_path: str,
    batch_size: int=16,
    x_seq_size: int=4,
    y_seq_size: int=20,
    vae_setup: bool=False,
    world_size: int=1,
    rank: int=0
) -> dl.KNMIDataModule:
    """ Setup the data module for training
    
    Args:
        train_IDs_fn (str): Path to the numpy file containing the training data IDs
        val_IDs_fn (str): Path to the numpy file containing the validation data IDs
    
    Returns:
        Data module for training
    """

    train_IDs = np.load(train_IDs_fn, allow_pickle=True) if train_IDs_fn is not None else None
    val_IDs = np.load(val_IDs_fn, allow_pickle=True) if val_IDs_fn is not None else None
    test_IDs = np.load(test_IDs_fn, allow_pickle=True) if test_IDs_fn is not None else None

    datamodule = dl.KNMIDataModule(
        train_data=train_IDs,
        val_data=val_IDs,
        test_data=test_IDs,
        data_path=data_path,
        batch_size=batch_size,
        x_seq_size=x_seq_size,
        y_seq_size=y_seq_size,
        vae_setup=vae_setup,
        world_size=world_size,
        rank=rank
    )
        
    return datamodule

def setup_model(
    device: torch.device,
    x_seq_size: int = 9,
    y_seq_size: int = 9,
    hidden_dim: int = 96,
    downscaling_factors: Tuple[int, int, int, int] = (4, 2, 2, 2),
    layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
    heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
    head_dim: int = 32,
    window_size: int = 9,
    relative_pos_embedding: bool = True,
    norm_dbz_values: bool = True,
    lr: float = 1e-4,
    rank: int = 0,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> Tuple[torch.nn.Module, optim.Optimizer, torch.nn.Module, optim.lr_scheduler.ReduceLROnPlateau]:
    
    net = Net(
        input_channel=x_seq_size,
        output_channel=y_seq_size,
        hidden_dim=hidden_dim,
        downscaling_factors=downscaling_factors,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        window_size=window_size,
        relative_pos_embedding=relative_pos_embedding,
        input_h_w=[256, 256]
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    criterion = BMAEloss(norm_dbz_values=norm_dbz_values).to(device)

    if rank == 0 and run:
        hyperparam_conf = {
                "optimizer_type": optimizer.__class__.__name__,
                "optimizer_betas": optimizer.defaults['betas'],
                "optimizer_weight_decay": optimizer.defaults['weight_decay'],
                "scheduler_type": lr_scheduler.__class__.__name__,
                "scheduler_patience": lr_scheduler.patience,
                "scheduler_factor": lr_scheduler.factor,
                "criterion": criterion.__class__.__name__,
            }
        run.config.update(hyperparam_conf)

    return net, optimizer, criterion, lr_scheduler


def load_checkpoint(ckpt_path, model_dir, device):
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            print(f"Resuming from given checkpoint: {ckpt_path}")
            return torch.load(ckpt_path, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    latest_ckpt_path = os.path.join(model_dir, "last.ckpt")
    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from latest checkpoint: {latest_ckpt_path}")
        return torch.load(latest_ckpt_path, map_location=device)
    
    print("No checkpoint found, starting from scratch")
    return None


def test_dataloading_time(datamodule, rank):
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    start = time.time()
    counter = 0
    for batch in tqdm(train_loader, desc="Training dataloading"):
        counter += 1
        if counter % 10 == 0:
            sys.stdout.flush()
    print("Time taken for training dataloading:", time.time()-start)

    start = time.time()
    counter = 0
    for batch in tqdm(val_loader, desc="Validation dataloading", disable=rank!=0):
        counter += 1
        if counter % 10 == 0:
            sys.stdout.flush()
    print("Time taken for validation dataloading:", time.time()-start)

def train_model(
        net, 
        epoch_size, 
        datamodule, 
        optimizer, 
        criterion, 
        lr_scheduler, 
        device, 
        start_epoch=0,
        model_dir=None,
        early_stopping_patience=5,
        world_size=1,
        rank=0,
        run=None,
        early_stopping=True  # New parameter
    ):
    datamodule.setup(stage='fit')
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()

    ckpt_saver = CheckpointSaver(ddp=world_size>1, model_dir=model_dir, k_best=3, save_all=True)

    best_val_loss = float("inf")
    epochs_no_improvement = 0
    early_stopping_signal = torch.tensor(False, dtype=torch.bool, device=device)

    for epoch in range(start_epoch, epoch_size):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_l_sum, test_l_sum, n_train = 0.0, 0.0, 0
        net.train()
    
        for batch in train_loader:
            # x, y = data_2_cnn(train_data, batch, batch_size, train_seq)
            (x,y) = batch
            while isinstance(x, list) or isinstance(x, tuple):
                x = x[0][0]
            
            x = x.squeeze().to(device)
            y = y.squeeze().to(device)

            y_hat = net(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_num = loss.detach().cpu().numpy()
            train_l_sum += loss_num
            n_train += x.size(0)

        train_l_sum, n_train = torch.tensor(train_l_sum, device=device), torch.tensor(n_train, device=device)
        dist.reduce(train_l_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_train, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss = (train_l_sum / n_train).item()
        
        dist.barrier()

        n_val = 0
        net.eval()
        with torch.no_grad():
            for batch in val_loader:
                # x, y = data_2_cnn(train_data, batch, batch_size, val_seq)
                (x,y) = batch
                while isinstance(x, list) or isinstance(x, tuple):
                    x = x[0][0]
                x = x.squeeze()
                y = y.squeeze()

                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)

                loss = criterion(y_hat, y)
                loss_num = loss.detach().cpu().numpy()
                test_l_sum += loss_num
                n_val += x.size(0)

            # test_loss = test_l_sum / n
            # lr_scheduler.step(test_loss)

            #TODO: ensure that remaining data from distributedsampler is also included in the loss
            test_l_sum, n_val = torch.tensor(test_l_sum, device=device), torch.tensor(n_val, device=device)
            dist.all_reduce(test_l_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_val, op=dist.ReduceOp.SUM)

            test_loss = (test_l_sum / n_val).item()
            last_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(test_loss)

            if rank == 0:
                print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                ckpt_saver.save_checkpoint(net, optimizer, lr_scheduler, epoch, test_loss)
                print('Train loss:', train_loss, ' Test loss:', test_loss)
                print('==='*20)

                if run:
                    run.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': test_loss,
                        'learning_rate': last_lr
                    })

                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1

                # Stop training if no improvement for specified patience
                if early_stopping and epochs_no_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    early_stopping_signal = torch.tensor(True, dtype=torch.bool, device=device)

        dist.broadcast(early_stopping_signal, src=0)
        if early_stopping_signal.item():
            break
                    
        
        # dist.barrier()
        
        # if world_size > 1:
        #     check_ddp_model_synchronization(net, rank)
 
        # if rank == 0:
        #     wandb.log({
        #         'epoch': epoch,
        #         'train_loss': train_loss,
        #         'val_loss': test_loss,
        #         'learning_rate': optimizer.param_groups[0]['lr']
        #     })
        
        # print('Train loss:', train_loss, ' Test loss:', test_loss)
        # print('==='*20)

        
def setup_and_train(
    rank,
    world_size,
    x_seq_size=4,
    y_seq_size=20,
    window_size=8,
    epoch_size=200,
    base_batch_size=16,
    norm_dbz_values=True,
    train_IDs_fn="list_IDs_200909_avg001mm_9x9y_rainformer_initialtest.npy",
    val_IDs_fn="list_IDs_200909_avg001mm_9x9y_rainformer_initialtest.npy",
    data_path=os.getenv('PREPROCESSED_RTCOR_DIR'),
    ckpt_path=None,
    model_dir=None,
    lr=1e-4,
    run=None,
    lr_orig=1e-4,
    batch_size_orig=16,
    y_seq_size_orig=9,
    early_stopping=True  # New parameter
):
    if rank == 0:
        print(f"y_seq_size: {y_seq_size}, type: {type(y_seq_size)}")
        print(f"learning rate: {lr}, type: {type(lr)}")
        print(f"batch size: {base_batch_size}, type: {type(base_batch_size)}")
        print(f"model dir: {model_dir}")

    setup_distributed(rank, world_size) # rank, world_size

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    datamodule = setup_data_module(
        train_IDs_fn=train_IDs_fn,
        val_IDs_fn=val_IDs_fn,
        test_IDs_fn=None,
        data_path=data_path,
        batch_size=base_batch_size,
        x_seq_size=x_seq_size,
        y_seq_size=y_seq_size,
        vae_setup=False,
        world_size=world_size,
        rank=rank
    )

    # test_dataloading_time(datamodule, rank)

    if rank == 0 and run:
        log_missing_hyperparams(setup_model, run=run, ignore=['device', 'world_size', 'rank'])

    lr = adjust_learning_rate(lr_orig, batch_size_orig, base_batch_size, world_size, y_seq_size_orig, y_seq_size)
    
    print(f"Adjusted learning rate: {lr:.8f}")

    net, optimizer, criterion, lr_scheduler = setup_model(rank=rank, lr=lr, device=device, x_seq_size=x_seq_size, y_seq_size=y_seq_size, window_size=window_size, norm_dbz_values=norm_dbz_values, run=run)
    start_epoch = 0

    if rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    dist.barrier()
    checkpoint = load_checkpoint(ckpt_path, model_dir, device)
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    dist.barrier()

    if world_size > 1:
        net = DDP(net, device_ids=[device])

    if rank == 0 and run:
        run.watch(net, log='all')
    
    train_model(net, epoch_size, datamodule, optimizer, criterion, lr_scheduler, device, start_epoch=start_epoch, model_dir=model_dir, world_size=world_size, rank=rank, run=run, early_stopping=early_stopping)

    cleanup()
    if run:
        run.finish()

    if rank == 0:
        print("\n\nDistributed training finished\n\n", flush=True)

def main(config=None, **kwargs):
    load_dotenv(dotenv_path="../../.env")

    config = OmegaConf.to_container(OmegaConf.load(config), resolve=True) if (config is not None) else {}
    config.update()
    config.update(kwargs)
    # print(f"Resolved model_dir: {config.get('model_dir')}")
    seed = 55
    set_seeds(seed, reproducible=config.get('reproducible', False))

    # world_size = int(os.getenv('SLURM_NTASKS'))  # Total number of tasks (processes)
    # rank = int(os.getenv('SLURM_PROCID'))  # Get rank (unique ID for each process)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    config['world_size'] = world_size
    config['rank'] = rank

    use_wandb = os.getenv('WANDB_API_KEY') is not None

    if use_wandb and rank == 0:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb_dir = os.path.join(os.getenv('RESULTS_DIR'), 'wandb')
        run = wandb.init(
            project='thesis-forecasting', 
            name=os.getenv('SLURM_JOB_NAME', 'rainformer-ddp'), 
            id=os.getenv('SLURM_JOB_NAME', None), 
            config=config, 
            resume='allow', 
            config_exclude_keys=['rank', 'device'],
            dir=wandb_dir
        )
        log_missing_hyperparams(setup_and_train, run)
    else:
        run = None

    setup_and_train(**config, run=run)


if __name__ == '__main__':
    Fire(main)

