import torch
import torch.nn.functional as F
from core.controllers import ControlMechanism


def _diagnose_layerwise_mismatches(pairs, tol, label):
    """Compares paired tensors layer-by-layer with a flat absolute tolerance.

    Args:
        pairs (Iterable[Tuple[torch.Tensor, torch.Tensor]]): Iterable of (a, b)
            tensor pairs, typically one pair per layer.
        tol (float): Maximum allowed element-wise absolute difference (|a - b|).
        label (str): Header description displayed at the top of any failure message.

    Raises:
        AssertionError: If any layer pair exceeds the specified absolute tolerance.

    Returns:
        None.
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
            f"  layers failing : {len(failing)}/{len(pairs)}\n" + "\n".join(failing)
        )


def _prime_and_forward(network, x, dendritic_effect):
    """Prime a network and compute a zero-control forward pass for validation.

    Sets the specified dendritic mode across all network populations, executes a
    mandatory baseline pass in evaluation mode to save state, and performs one
    instantaneous controlled forward pass with zeroed controls.

    Args:
        network (torch.nn.Module): The target neural network module containing
            populations.
        x (torch.Tensor): The input batch tensor fed into the network.
        dendritic_effect (Any): The dendritic mode or setting to assign to each
            neural population.

    Returns:
        torch.Tensor: The output tensor produced by the zero-control forward pass.
    """
    for pop in network.populations:
        pop.dendritic_effect = dendritic_effect

    controller = ControlMechanism(mode="backprop")
    zeros = controller.initialize_controls(x.size(0), network.populations)

    with torch.no_grad():
        network.eval()
        network.forward(x, control_signals=None, save_baseline=True)
        network.train()

    return network.forward(
        x, control_signals=zeros, save_baseline=False, dynamic_step=False
    )


def _autograd_grads(network, y, global_control):
    """Compute the autograd gradients of each population's pre-activation state.

    Retains gradients for each population's `z` tensor, backpropagates through
    `(global_control * y).sum()`, and returns the cloned gradient values.

    Args:
        network (torch.nn.Module): The neural network whose layer activations are
            being inspected.
        y (torch.Tensor): Output tensor from a prior forward pass.
        global_control (torch.Tensor): Global control tensor multiplied by `y` before
            backpropagation.

    Returns:
        list[torch.Tensor]: A gradient tensor for each population's `z` state.
    """
    for pop in network.populations:
        if pop.z is not None:
            pop.z.retain_grad()
    (global_control * y).sum().backward()
    return [pop.z.grad.clone() for pop in network.populations]


def _cosine_sims(tensor_a, tensor_b):
    """Calculates per-layer cosine similarity between two sequences of tensors.

    Args:
        tensor_a (Iterable[torch.Tensor]): The first sequence of layer tensors.
        tensor_b (Iterable[torch.Tensor]): The second sequence of layer tensors.

    Returns:
        List[float]: A list of scalar cosine similarity values for each matching pair.
    """
    return [
        F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        for a, b in zip(tensor_a, tensor_b)
    ]
