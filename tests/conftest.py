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

# Let pytest run this before every test in scope, whether the test asks for it or not. 
@pytest.fixture(autouse=True)
def deterministic():
    """Seed everything before every test for reproducibility."""
    torch.manual_seed(0)

@pytest.fixture
def tiny_network():
    """A 2-4-1 network, freshly initialized, on CPU."""
    return Network(pop_sizes=[2, 4, 1])

def tiny_additive_network():
    """A 2-4-1 network, with linear dendritic modulation, freshly initialized, on CPU."""
    return Network(pop_sizes=[2, 4, 1], dendritic_effect="additive")

@pytest.fixture
def mnist_network():
    """A small MNIST-shaped network: 784 -> 32 -> 10.

    Real MNIST has a 10-dim output, which exposes the GN-vs-BP distinction
    that doesn't show up in the 1-D XOR network. Useful for tests that
    need to probe the multi-output regime of DFC theory.
    """
    return Network(pop_sizes=[784, 32, 10])

@pytest.fixture
def tiny_batch():
    """A 4-sample batch matching XOR shape."""
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return x, y

@pytest.fixture
def mnist_tiny_batch():
    """A 4-sample fake-MNIST batch: random inputs, one-hot labels.
    """
    torch.manual_seed(42)
    x = torch.randn(4, 784) * 0.3
    y = torch.zeros(4, 10)
    y[range(4), torch.tensor([0, 1, 2, 3])] = 1.0
    return x, y