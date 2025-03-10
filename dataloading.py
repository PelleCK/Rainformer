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

    def __init__(self, list_IDs, data_path, x_seq_size=6, y_seq_size=3, h_w=(256, 256), vae_setup=True, load_prep=True):
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size
        self.h_w = h_w
        self.load_prep = load_prep
        self.vae_setup = vae_setup

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.x_seq_size, *self.h_w, 1), dtype=np.float32)
        y = np.empty((self.y_seq_size, *self.h_w, 1), dtype=np.float32)
        
        x_IDs, y_IDs = list_IDs_temp
        for t in range(self.x_seq_size):
            try:
                X[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(x_IDs[t])), mmap_mode='r')
            except (FileNotFoundError, EOFError) as e:
                print(f"Error loading file: {x_IDs[t]}", flush=True)
                print(f"Shape of x_IDs: {len(x_IDs)}", flush=True)
                print(f"Expected x_seq_size: {self.x_seq_size}", flush=True)
                print(f"Exception: {e}", flush=True)
                
        if self.vae_setup:
            y = np.copy(X)
        else:
            for t in range(self.y_seq_size):
                try:
                    y[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(y_IDs[t])), mmap_mode='r')
                except (FileNotFoundError, EOFError) as e:
                    print(f"Error loading file: {y_IDs[t]}", flush=True)
                    print(f"Shape of y_IDs: {len(y_IDs)}", flush=True)
                    print(f"Expected y_seq_size: {self.y_seq_size}", flush=True)
                    print(f"Exception: {e}", flush=True)

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
    def __init__(self, train_data, val_data, test_data, data_path, batch_size=32, x_seq_size=6, y_seq_size=3, h_w=(256, 256), vae_setup=True, world_size=1, rank=0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.data_path = data_path
        self.batch_size = batch_size
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size
        self.h_w = h_w
        self.vae_setup = vae_setup
        self.world_size = world_size
        self.rank = rank

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = KNMIDataset(self.train_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, h_w=self.h_w, vae_setup=self.vae_setup)
            self.val_dataset = KNMIDataset(self.val_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, h_w=self.h_w, vae_setup=self.vae_setup)
        elif stage == 'test':
            self.test_dataset = KNMIDataset(self.test_data, self.data_path, 
                                            x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size, h_w=self.h_w, vae_setup=self.vae_setup)

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
            drop_last=True,
            sampler=sampler,
            prefetch_factor=4,
            persistent_workers=False
        )
    



class OriginalDataSet(Dataset):
    def __init__(self, test_seq_path, test_data, batch_size):
        """
        Custom dataset for handling original test data loading.

        Args:
            test_seq_path (str): Path to the test_seq.npy file.
            test_data (np.array): Test data containing sequences.
            batch_size (int): Batch size for splitting indices.
        """
        self.test_seq = np.load(test_seq_path)  # Load test_seq.npy
        self.test_data = test_data             # Test data from h5 file
        self.batch_size = batch_size
        self.batches = self._create_batches()  # Create batch index ranges

    def _create_batches(self):
        """
        Create the list of batch ranges based on test_seq and batch_size.
        """
        ran = np.arange(self.batch_size, self.test_seq.shape[0], self.batch_size)
        batches = []
        start_idx = 0
        for end_idx in ran:
            batches.append((start_idx, end_idx))
            start_idx = end_idx
        if start_idx < self.test_seq.shape[0]:
            batches.append((start_idx, self.test_seq.shape[0]))  # Add remainder batch
        return batches

    def __len__(self):
        """
        Returns the total number of batches.
        """
        return len(self.batches)

    def data_2_cnn(self, data, indices):
        """
        Function that transforms data based on given indices.

        Args:
            data (np.array): The full test data.
            indices (list): List of sequence indices to retrieve.

        Returns:
            torch.Tensor: x and y tensors ready for the model.
        """
        result = []
        for i in indices:
            tmp = data[i] * 4783 / 100 * 12  # Apply denormalisation
            result.append(torch.tensor(tmp, dtype=torch.float))
        result = torch.stack(result, dim=0)
        x = result[:, :9]  # Input sequence
        y = result[:, 9:]  # Output sequence
        return x, y

    def __getitem__(self, idx):
        """
        Retrieves a batch of data based on index.

        Args:
            idx (int): Batch index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: x (input), y (target) tensors.
        """
        start, end = self.batches[idx]
        sequence_indices = self.test_seq[start:end]
        x, y = self.data_2_cnn(self.test_data, sequence_indices)
        return x, y


def get_original_dataloader(test_seq_path, test_data, batch_size):
    """
    Function to initialize the dataloader.

    Args:
        test_seq_path (str): Path to the test_seq.npy file.
        test_data (np.array): Test data containing sequences.
        batch_size (int): Batch size for evaluation.

    Returns:
        DataLoader: Torch DataLoader instance.
    """
    dataset = OriginalDataSet(test_seq_path, test_data, batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch size 1 because there already are batches in dataset
    return dataloader
