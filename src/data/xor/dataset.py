import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_logger

logger = get_logger()


class XORDataset(Dataset):  # build a custom class from torch's Dataset class
    """XORDataset."""

    def __init__(self):
        """Initialize the XOR dataset with all four binary input-output pairs.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()
        # Inputs (X): All combinations of binary inputs for XOR
        self.X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        # Targets (Y): The Exclusive OR logic, ordered to match the idx of the input combinations
        self.Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    def __len__(self):
        """Return the number of samples available in the XOR dataset.

        Args:
            None.

        Returns:
            int: Total number of XOR samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """Retrieve one XOR sample and its corresponding target.

        Args:
            idx (int): Index of the requested sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input pair and binary XOR target.
        """
        return self.X[idx], self.Y[idx]


def get_dataloader(batch_size=4, shuffle=True):
    """Create a DataLoader for the XOR dataset.

    Args:
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle sample order each epoch.

    Returns:
        DataLoader: DataLoader wrapping the XOR dataset.
    """
    dataset = XORDataset()  # create an instance of the dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
