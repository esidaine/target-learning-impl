import pytest
import torch
from models.network import Network


"""
pytest                          # run everything
pytest tests/test_network.py    # one file
pytest -k "local_controls"      # any test name matching pattern
pytest -x                       # stop at first failure (great while debugging)
pytest --lf                     # only re-run tests that failed last time
"""

@pytest.fixture(autouse=True)
def deterministic():
    """Seed everything before every test for reproducibility."""
    torch.manual_seed(0)

@pytest.fixture
def tiny_network():
    """A 2-4-1 network, freshly initialized, on CPU."""
    return Network(pop_sizes=[2, 4, 1])

@pytest.fixture
def tiny_batch():
    """A 4-sample batch matching XOR shape."""
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return x, y