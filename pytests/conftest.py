import logging
import pytest
import torch
from data.xor.dataset import get_dataloader
from models.network import Network

logging.getLogger().setLevel(logging.DEBUG)

"""
pytest                          # run everything
pytest tests/test_network.py    # one file
pytest -k "local_controls"      # any test name matching pattern
pytest -x                       # stop at first failure (great while debugging)
pytest --lf                     # only re-run tests that failed last time
"""


# Let pytest run this before every test in scope, whether the test asks for it or not.
@pytest.fixture(autouse=True)
def deterministic():
    """Seed random number generators before every test for reproducibility.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)


@pytest.fixture
def tiny_network():
    """Creates a freshly initialized 2-4-1 Network model on CPU.

    Args:
        None.

    Returns:
        Network: A network instance configured with population sizes [2, 4, 1].
    """
    return Network(pop_sizes=[2, 4, 1])


@pytest.fixture
def tiny_additive_network():
    """Creates a 2-4-1 Network model with additive dendritic modulation.

    Args:
        None.

    Returns:
        Network: A network instance configured with additive dendritic effects
        and population sizes [2, 4, 1].
    """
    return Network(pop_sizes=[2, 4, 1], dendritic_effect="additive")


@pytest.fixture
def mnist_network():
    """Creates a multi-output Network model (784 -> 32 -> 10) for MNIST experiments.

    Exposes the distinction between Gauss-Newton (GN) and Backpropagation (BP)
    dynamics that does not show up in 1-D output networks.

    Args:
        None.

    Returns:
        Network: A network instance configured with population sizes [784, 32, 10].
    """
    return Network(pop_sizes=[784, 32, 10])


@pytest.fixture
def tiny_batch():
    """Generates a 4-sample synthetic batch matching XOR inputs and targets.

    Args:
        None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (x, y) containing input features
        `x` of shape (4, 2) and target labels `y` of shape (4, 1).
    """
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    return x, y


@pytest.fixture
def mnist_tiny_batch():
    """Generates a 4-sample synthetic MNIST batch with random inputs and one-hot targets.

    Args:
        None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (x, y) containing input features
        `x` of shape (4, 784) and target labels `y` of shape (4, 10).
    """
    torch.manual_seed(42)
    x = torch.randn(4, 784) * 0.3
    y = torch.zeros(4, 10)
    y[range(4), torch.tensor([0, 1, 2, 3])] = 1.0
    return x, y


@pytest.fixture
def xor_dataloader():
    """Provides a fixed-order DataLoader for XOR dataset evaluations.

    Args:
        None.

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader configured with batch
        size 4 and shuffling disabled for deterministic sequencing.
    """
    return get_dataloader(batch_size=4, shuffle=False)
