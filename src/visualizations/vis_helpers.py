"""Helpers for exporting controller state to visualization tools."""

from typing import Any

import numpy as np


def make_manim_snapshot(network: Any, local_controls: list[Any], metrics: Any) -> None:
    """Record the current activation state for a later Manim animation.

    The controller calls this once per PID step. Keeping snapshots as NumPy
    arrays makes them independent of autograd and safe to serialize with
    ``pickle`` after training.
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