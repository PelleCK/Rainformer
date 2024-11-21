import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class KNMIDataset(Dataset):
    """Custom dataset for loading preprocessed KNMI data

    Args:
        list_IDs (list): List of tuples, each containing sequences of input and target filenames.
        data_path (str): Path to the directory containing the preprocessed data files.
        x_seq_size (int): Number of timesteps in the input sequence.
        y_seq_size (int): Number of timesteps in the target sequence.
        load_prep (bool): Flag indicating whether the data is preprocessed.
    """

    def __init__(self, list_IDs, data_path, x_seq_size=6, y_seq_size=3, vae_setup=True, load_prep=True):
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size
        self.load_prep = load_prep
        self.vae_setup = vae_setup

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.x_seq_size, 256, 256, 1), dtype=np.float32)
        y = np.empty((self.y_seq_size, 256, 256, 1), dtype=np.float32)
        
        x_IDs, y_IDs = list_IDs_temp
        for t in range(self.x_seq_size):
            X[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(x_IDs[t])), mmap_mode='r')

        if self.vae_setup:
            y = np.copy(X)
        else:
            for t in range(self.y_seq_size):
                y[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(y_IDs[t])), mmap_mode='r')

        # Permute to match the required shape: (x_seq_size, img_width, img_height, 1) to (1, 1, x_seq_size, img_width, img_height)
        X = torch.tensor(X).permute(3, 0, 1, 2).unsqueeze(0)
        y = torch.tensor(y).permute(3, 0, 1, 2).unsqueeze(0)

        return X, y

# Custom collate function for batching the data
# This ensures that each batch of KNMI data is in the right shape for the model
def collate_fn(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    data = torch.stack(data).squeeze(dim=1)  # (batch_size, 1, 1, X_sequence_length, 256, 256)
    targets = torch.stack(targets).squeeze(dim=1)  # (batch_size, 1, Y_sequence_length, 256, 256)

    # Create timestamps tensor for the batch
    batch_size = data.size(0)
    x_seq_size = data.size(2)
    timestamps = torch.arange(x_seq_size).float().unsqueeze(0).repeat(batch_size, 1)  # (batch_size, X_sequence_length)

    return ([[data, timestamps]], targets)

class KNMIDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for loading KNMI data

    Args:
        train_data (list): List of training data file IDs.
        val_data (list): List of validation data file IDs.
        data_path (str): Path to the directory containing the preprocessed data files.
        batch_size (int): Batch size for the DataLoader.
        x_seq_size (int): Number of timesteps in the input sequence.
        y_seq_size (int): Number of timesteps in the target sequence.
    """
    def __init__(self, train_data, val_data, test_data, data_path, batch_size=32, x_seq_size=6, y_seq_size=3, vae_setup=True, world_size=1, rank=0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.data_path = data_path
        self.batch_size = batch_size
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size
        self.vae_setup = vae_setup
        self.world_size = world_size
        self.rank = rank

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = KNMIDataset(self.train_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, vae_setup=self.vae_setup)
            self.val_dataset = KNMIDataset(self.val_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, vae_setup=self.vae_setup)
        elif stage == 'test':
            self.test_dataset = KNMIDataset(self.test_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, vae_setup=self.vae_setup)

    def train_dataloader(self):
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=(sampler is None), 
            num_workers=3, 
            pin_memory=True, 
            collate_fn=collate_fn,
            drop_last=False,
            sampler=sampler,
            prefetch_factor=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=True)
        
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=3, 
            pin_memory=True, 
            collate_fn=collate_fn,
            drop_last=False,
            sampler=sampler,
            prefetch_factor=4,
            persistent_workers=True
        )
    #TODO: add test dataloader

    def test_dataloader(self):
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(self.test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)
        
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=3, 
            pin_memory=True, 
            collate_fn=collate_fn,
            drop_last=False,
            sampler=sampler,
            prefetch_factor=4,
            persistent_workers=False
        )