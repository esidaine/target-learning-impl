import torch
import torch.nn.functional as F
from core.controllers import ControlMechanism

def _diagnose_layerwise_mismatches(pairs, tol, label):
    """
    Compare paired tensors layer-by-layer with a flat absolute tolerance.
    Raises AssertionError with a per-layer diagnostic breakdown on mismatch.

    pairs : iterable of (a, b) tensor pairs, one per layer
    tol   : maximum allowed |a - b| element-wise
    label : description shown at the top of any failure message
    """
    pairs = list(pairs)
    failing = []
    for i, (a, b) in enumerate(pairs):
        diff = (a - b).abs()
        max_d = diff.max().item()
        if max_d <= tol:
            continue
        mean_d = diff.mean().item()
        violating = (diff > tol).float().mean().item()
        kind = "systematic" if max_d < 3 * mean_d else "outlier"
        failing.append(
            f"  L{i}: max={max_d:.2e}  mean={mean_d:.2e}  "
            f"violating={violating:.1%}  ({kind})"
        )

    if failing:
        raise AssertionError(
            f"\n{label}\n"
            f"  tolerance      : {tol:.0e}\n"
            f"  layers failing : {len(failing)}/{len(pairs)}\n"
            + "\n".join(failing)
        )


def _prime_and_forward(network, x, dendritic_effect):
    """
    Set dendritic mode, run the mandatory baseline pass, then return y from
    an instantaneous (dynamic_step=False) controlled forward pass with zero
    controls.  Using dynamic_step=False gives a clean autograd graph:
    y = f(z), so pop.z.grad from backward() is exactly f'(z) * upstream.
    """
    for pop in network.populations:
        pop.dendritic_effect = dendritic_effect

    controller = ControlMechanism(mode='backprop')
    zeros = controller.initialize_controls(x.size(0), network.populations)

    with torch.no_grad():
        network.eval()
        network.forward(x, control_signals=None, save_baseline=True)
        network.train()

    return network.forward(x, control_signals=zeros,
                           save_baseline=False, dynamic_step=False)


def _autograd_grads(network, y, global_control):
    """
    Compute d/dz_i [(u * y).sum()] via autograd and return the per-layer
    list.  Must be called AFTER any method that reads pop.z, because
    backward() releases the computation graph.
    """
    for pop in network.populations:
        if pop.z is not None:
            pop.z.retain_grad()
    (global_control * y).sum().backward()
    return [pop.z.grad.clone() for pop in network.populations]


def _cosine_sims(tensor_a, tensor_b):
    """Per-layer cosine similarity between two lists of tensors."""
    return [
        F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        for a, b in zip(tensor_a, tensor_b)
    ]