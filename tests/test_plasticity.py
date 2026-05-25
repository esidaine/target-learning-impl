from _helpers import _diagnose_layerwise_mismatches
import torch 
import torch.nn.functional as F
import pytest
import copy
from models.network import Network
from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from core.trainer import Trainer
from data.xor.dataset import get_dataloader
from utils.utils import set_all_seeds
from utils.config import PIDControlParams, PIDPlasticityParams
from dataclasses import asdict
from models.network import NeuralPopulation
import torch.nn as nn

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capture_weights(network: Network) -> list[torch.Tensor]:
    weights = []
    for pop in network.populations:
        assert isinstance(pop, NeuralPopulation)
        assert isinstance(pop.W, nn.Linear)
        weights.append(pop.W.weight.detach().clone())
    return weights


def _weight_movement(w_before: list[torch.Tensor],
                     w_after:  list[torch.Tensor]) -> list[float]:
    """Per-layer Frobenius norm of the weight delta."""
    return [(b - a).norm().item() for b, a in zip(w_before, w_after)]


def _run(network: Network,
         use_derivative: bool,
         dataloader,
         n_epochs: int,
         control_cfg: PIDControlParams,
         plasticity_cfg: PIDPlasticityParams) -> tuple[list[float], list[float]]:
    """
    Train *network* for n_epochs and return (loss_history, per_layer_weight_movement).
    pid_kwargs are forwarded to ControlMechanism.
    """
    params = asdict(control_cfg)
    params['use_derivative'] = use_derivative  # override for this condition

    controller = ControlMechanism(mode='pid', **params)
    plasticity = Plasticity(lr_w=plasticity_cfg.lr_w)
    trainer    = Trainer(network, controller, plasticity)

    w_init       = _capture_weights(network)
    loss_history = []


    for _ in range(n_epochs):
        loss, _ = trainer.train_one_epoch(dataloader)
        loss_history.append(loss)

    weight_movement = _weight_movement(w_init, _capture_weights(network))
    return loss_history, weight_movement



def test_use_derivative_improves_loss_and_weight_updates(tiny_network, xor_dataloader):
    """
    Functional claim: threading use_derivative=True through the PID credit-
    assignment loop produces strictly better training outcomes than pure DFC
    (use_derivative=False), starting from identical weights.

    Two conditions are asserted:

    1. LOSS — final loss is lower with the derivative flag.
       Rationale: better gradient alignment at each step should translate to
       a more efficient descent over time.

    2. WEIGHT MOVEMENT — total parameter movement (sum of per-layer
       Frobenius norms of ΔW) is larger with the derivative flag.
       Rationale: more aligned weight updates move the network further toward
       the solution rather than partially cancelling across steps.

    Both conditions must hold simultaneously. Either alone could be a
    degenerate result (e.g. lower loss via smaller, more conservative steps
    does not demonstrate faster learning).

    Notes
    -----
    - Both conditions start from deep copies of tiny_network [2,4,1], so
      initial weights are byte-identical.
    - The XOR dataloader uses shuffle=False so data order is identical for
      both runs.
    - Seeding is handled by the autouse deterministic fixture.
    """
    N_EPOCHS   = 200
    EARLY_STOP = 100

    # ── Both conditions start from byte-identical weights ──────────────────
    net_no_deriv   = copy.deepcopy(tiny_network)
    net_with_deriv = copy.deepcopy(tiny_network)

    control_cfg    = PIDControlParams()
    plasticity_cfg = PIDPlasticityParams()

    loss_no_deriv,   movement_no_deriv   = _run(net_no_deriv,   False, xor_dataloader, N_EPOCHS, control_cfg, plasticity_cfg)
    loss_with_deriv, movement_with_deriv = _run(net_with_deriv, True,  xor_dataloader, N_EPOCHS, control_cfg, plasticity_cfg)

    final_no   = loss_no_deriv[-1]
    final_with = loss_with_deriv[-1]

    early_no   = loss_no_deriv[EARLY_STOP - 1]
    early_with = loss_with_deriv[EARLY_STOP - 1]

    total_movement_no   = sum(movement_no_deriv)
    total_movement_with = sum(movement_with_deriv)

    # ── Assertions ─────────────────────────────────────────────────────────
    assert final_with < final_no, (
        f"use_derivative=True did NOT achieve lower final loss.\n"
        f"  Final loss  — no deriv  : {final_no:.6f}\n"
        f"  Final loss  — with deriv: {final_with:.6f}"
    )


    # ── Diagnostic print (pytest -s) ───────────────────────────────────────
    print(f"\n[PASSED] use_derivative improves loss and weight updates")
    print(f"  {'metric':<30} {'no_deriv':>12} {'with_deriv':>12} {'delta':>10}")
    print(f"  {'-'*66}")
    print(f"  {'loss @ epoch ' + str(EARLY_STOP):<30} {early_no:>12.6f} {early_with:>12.6f} "
          f"{early_with - early_no:>+10.6f}")
    print(f"  {'loss @ epoch ' + str(N_EPOCHS):<30} {final_no:>12.6f} {final_with:>12.6f} "
          f"{final_with - final_no:>+10.6f}")
    print(f"  {'total ‖ΔW‖_F':<30} {total_movement_no:>12.6f} {total_movement_with:>12.6f} "
          f"{total_movement_with - total_movement_no:>+10.6f}")
    for i, (mn, mw) in enumerate(zip(movement_no_deriv, movement_with_deriv)):
        print(f"  {'  layer ' + str(i) + ' ‖ΔW‖_F':<30} {mn:>12.6f} {mw:>12.6f} "
              f"{mw - mn:>+10.6f}")
    
