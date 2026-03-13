
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath('..')) # Append the absolute path of the project root to Python's search map to enable core imports
from src.core.utils import get_logger
logger = get_logger()

class XORDataset(Dataset): # build a custom class from torch's Dataset class
    def __init__(self):
        super().__init__() 
        # Inputs (X): All combinations of binary inputs for XOR
        self.X = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32) 
        # Targets (Y): The Exclusive OR logic, ordered to match the idx of the input combinations
        self.Y = torch.tensor([[0], 
                               [1], 
                               [1], 
                               [0]], dtype=torch.float32)
        
    def __len__(self):
        """Returns the total number of samples in the dataset for pytorch's Dataloader class."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Retrieves a single sample and its corresponding target."""
        return self.X[idx], self.Y[idx]
        
def get_dataloader(batch_size=4, shuffle=True):
    """ Creates the datset and calls the DataLoader which groups the samples into batches and shuffles them"""
    dataset = XORDataset() # create an instance of the dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

