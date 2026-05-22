from core.controllers import ControlMechanism, OptimizationMetrics
from core.plasticity import Plasticity
from _helpers import _diagnose_layerwise_mismatches
import torch 
import torch.nn.functional as F
import pytest

@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_zero_error_produces_zero_update(tiny_network, tiny_batch, mode):
    """If target == baseline_pred, nothing should change."""
    x, _ = tiny_batch
    controller = ControlMechanism(mode=mode, lr_c=0.1, max_steps=60)
    plasticity = Plasticity(lr_w=0.5)

    with torch.no_grad():
        baseline_pred = tiny_network(x, control_signals=None, save_baseline=True)
    target_y = baseline_pred.clone()

    weights_before = [pop.W.weight.detach().clone() for pop in tiny_network.populations]

    # Public API handles baseline recording, mode dispatch, and cleanup.
    controller.optimize_control_signal(x, target_y, tiny_network)
    plasticity.update_weights(tiny_network, x)

    weights_after = [pop.W.weight.detach().clone() for pop in tiny_network.populations]

    _diagnose_layerwise_mismatches(
        pairs=zip(weights_before, weights_after),
        tol=1e-10,
        label="weights changed after zero-error step",
    )
    _diagnose_layerwise_mismatches(
        pairs=[(pop.a_controlled, pop.a_baseline) for pop in tiny_network.populations],
        tol=1e-10,
        label="a_controlled vs a_baseline at zero error",
    )

def test_learning_rule(): 
    plasticity = Plasticity(lr_w=0.5)

    # dim is [batch_size, num_neurons]
    a_pre = torch.tensor(
                        [[1.0, 2.0], 
                        [3.0, 4.0]])
    a_baseline = torch.tensor(
                        [[0.0, 1.0],
                        [2.0, 0.0]])
    a_controlled = torch.tensor(
                        [[1.0, 0.0],
                        [0.0, 1.0]])

    # delta_W = torch.matmul(errors.T, a_pre)
    expected_delta_W = torch.tensor(
                        [[-5.0, -6.0],
                        [2.0, 2.0]])
    
    # divide by batch size to get average update
    expected_delta_W = expected_delta_W / a_pre.size(0)

    computed_delta_W, _ = plasticity.learning_rule(a_pre, a_baseline, a_controlled)
    assert torch.allclose(computed_delta_W, expected_delta_W)
    
