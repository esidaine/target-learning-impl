import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from utils.utils import get_logger

logger = get_logger()


class MNISTDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "./data",
        train: bool = True,
        flatten: bool = True,
        num_classes: int = 10,
        target_on: float = 1.0,
        target_off: float = 0.05,
        norm_mean: float = 0.1307,
        norm_std: float = 0.3081,
    ):
        super().__init__()
        self.train = train
        self.flatten = flatten
        self.num_classes = num_classes
        self.target_on = target_on
        self.target_off = target_off

        # 1. Group the transforms together using Compose.
        # 2. Convert to Tensor first
        # 3. Use the nrm_mean norm_std for mnist Z-score normalization, stops the raw input from blowing up the pre-synaptic summation and causing saturation of the activation function.
        # 4. Assign it to self.transform.
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((norm_mean,), (norm_std,))]
        )

        logger.info(f"Loading MNIST {'Train' if train else 'Test'} dataset...")
        self.dataset = MNIST(
            root=root_dir, train=train, download=True, transform=self.transform
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample and its corresponding continuous target.
        Target learning requires continuous target vectors rather than class indices.
        """
        x, y_idx = self.dataset[idx]

        if self.flatten:
            x = x.view(-1)  # Flattens 1x28x28 to 784

        # Create continuous target vector for the PID controller (One-Hot Encoding)
        # Not strcitly to 0 and 1, but keeps neurons slightly active to maintain gradient flow and prevent dead neurons.
        y_target = torch.full((self.num_classes,), self.target_off, dtype=torch.float32)
        y_target[y_idx] = self.target_on

        return x, y_target


def get_dataloader(
    batch_size: int = 64,
    train: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    flatten: bool = True,
    target_on: float = 1.0,
    target_off: float = 0.05,
    norm_mean: float = 0.1307,
    norm_std: float = 0.3081,
) -> DataLoader:
    """
    Creates the dataset and returns a PyTorch DataLoader configured for DFC experiments.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): Whether to load the training or test split.
        shuffle (Optional[bool]): Whether to shuffle the data. Defaults to True for train, False for test.
        num_workers (int): Number of subprocesses for data loading.
        flatten (bool): Whether to flatten the images to 1D vectors.
        target_on (float): Continuous target value for the correct class.
        target_off (float): Continuous baseline target value for incorrect classes (prevents dead neurons).
        norm_mean (float): Mean for Z-score normalization.
        norm_std (float): Standard deviation for Z-score normalization.

    Returns:
        DataLoader: PyTorch dataloader instance.
    """
    if shuffle is None:
        shuffle = (
            train  # Conventionally shuffle training data, but evaluate sequentially
        )

    # Pass the dynamical constraints down into the dataset
    dataset = MNISTDataset(
        train=train,
        flatten=flatten,
        target_on=target_on,
        target_off=target_off,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # logger  outputs the physical target bounds [On: 1.0, Off: 0.05] and other configurations
    logger.info(
        f"DataLoader init - Batch: {batch_size}, Train: {train}, Shuffle: {shuffle}, "
        f"Targets: [On: {target_on}, Off: {target_off}]"
    )

    # instantiate loaders like this:
    # get_dataloader(target_off=0.01) vs get_dataloader(target_off=-0.05)
    # to easily test how different resting baselines affect the stability

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU-to-GPU memory transfer
    )
