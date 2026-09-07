from pathlib import Path

import pytest
import torch

from data.mnist.dataset import get_dataloader
from models.network import Network
from utils.config import ExperimentConfig


def test_mnist_batch_is_ready_for_training():
    """A real MNIST batch has the shape and target format the network expects."""
    config = ExperimentConfig(task="mnist")
    batch_size = 8
    expected_input_features = config.pop_sizes[0]
    expected_output_features = config.pop_sizes[-1]
    data_root = Path(__file__).parents[1] / "data"
    train_images = data_root / "MNIST" / "raw" / "train-images-idx3-ubyte"
    if not train_images.exists():
        pytest.skip("MNIST is not downloaded locally")

    dataloader = get_dataloader(
        batch_size=batch_size,
        root_dir=str(data_root),
        train=True,
        shuffle=False,
        num_workers=0,
        num_classes=expected_output_features,
    )
    inputs, targets = next(iter(dataloader))

    assert inputs.shape == (batch_size, expected_input_features)
    assert targets.shape == (batch_size, expected_output_features)
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert torch.isfinite(inputs).all()
    assert torch.isfinite(targets).all()
    expected_target_sum = 1.0 + (expected_output_features - 1) * 0.05
    assert torch.allclose(
        targets.sum(dim=1),
        torch.full((batch_size,), expected_target_sum),
    )
    assert torch.all(targets.max(dim=1).values == 1.0)
    assert torch.all(targets.min(dim=1).values == 0.05)

    network = Network(pop_sizes=config.pop_sizes)
    predictions = network(
        inputs,
        control_signals=None,
        save_baseline=False,
    )
    assert predictions.shape == (8, 10)
    assert torch.isfinite(predictions).all()