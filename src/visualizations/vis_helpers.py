"""Helpers for exporting controller state to visualization tools."""

from typing import Any

import numpy as np


def make_manim_snapshot(network: Any, local_controls: list[Any], metrics: Any) -> None:
    """Record the current activation state for a later Manim animation.

    The controller calls this once per PI step. Keeping snapshots as NumPy
    arrays makes them independent of autograd and safe to serialize with
    ``pickle`` after training.

    Args:
        network (Any): Network object containing populations with controlled activations.
        local_controls (list[Any]): Per-layer local controls for the current step.
        metrics (Any): Metrics object that stores state-history snapshots.

    Returns:
        None.

    Raises:
        RuntimeError: If controlled activations are unavailable when called.
    """
    del local_controls

    state = []
    for population in network.populations:
        activation = population.a_controlled
        if activation is None:
            raise RuntimeError(
                "Cannot capture a visualization snapshot before a controlled forward pass."
            )
        state.append(activation.detach().cpu().numpy().copy())

    metrics.state_history.append(state)
