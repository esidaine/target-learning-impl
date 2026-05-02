import torch 

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