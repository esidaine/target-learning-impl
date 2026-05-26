import torch
from core.controllers import ControlMechanism
from torch.nn import functional as F
import pytest
from _helpers import _prime_and_forward, _autograd_grads, _cosine_sims
from utils.utils import set_all_seeds
from models.network import Network

@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_chain_rule_matches_autograd_exactly(tiny_network, tiny_batch, dendritic_effect):
    """
    Mathematical claim: chain_rule_project_feedback is exact backpropagation.

    For an instantaneous forward pass the recurrence
        Ψ_L = u * f'(z_L)
        Ψ_i = W_{i+1}^T Ψ_{i+1} * f'(z_i)
    is algebraically identical to d/dz_i [(u * y).sum()].

    Expected: cosine similarity > 0.999 on every layer (floating-point exact).
    A lower value indicates a bug in either the SiLU derivative or the
    chain-rule loop.
    """
    torch.manual_seed(42)
    x, _ = tiny_batch
    x = x.detach().requires_grad_(True)

    y = _prime_and_forward(tiny_network, x, dendritic_effect)
    u = torch.randn_like(y)

    # Read pop.z BEFORE backward() releases the graph
    chain_controls = tiny_network.chain_rule_project_feedback(u)
    autograd_grads = _autograd_grads(tiny_network, y, u)

    sims = _cosine_sims(chain_controls, autograd_grads)
    worst = min(sims)

    assert worst > 0.999, (
        f"chain_rule_project_feedback diverges from autograd "
        f"[dendritic_effect='{dendritic_effect}'].\n"
        f"  Per-layer cos_sims : {[f'{s:.6f}' for s in sims]}\n"
        f"  Worst layer        : {worst:.6f}\n"
        f"  Check: SiLU derivative formula and W^T indexing in chain loop."
    )


@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_dfc_feedback_positively_aligned_with_gradient(tiny_network, tiny_batch, dendritic_effect):
    """
    Theoretical claim: DFC with Q initialised to J^T is gradient-aligned.

    Pure DFC (use_derivative=False): c_i = Q_i u = W_i^T u.
    With Q_i = J_i^T at init, the feedback approximates the gradient
    direction without the chain rule.  The claim is weak: cos_sim > 0
    (positive alignment on every layer).  This is the minimum condition
    for DFC to act as a useful learning signal.

    A failure here points to a bug in the Q initialisation block of Network.
    """
    torch.manual_seed(42)
    x, _ = tiny_batch
    x = x.detach().requires_grad_(True)

    for pop in tiny_network.populations:
        pop.dendritic_effect = dendritic_effect

    y = _prime_and_forward(tiny_network, x, dendritic_effect)
    u = torch.randn_like(y)

    dfc_controls = tiny_network.DFC_project_feedback(u, use_derivative=False)
    autograd_grads = _autograd_grads(tiny_network, y, u)

    sims = _cosine_sims(dfc_controls, autograd_grads)
    worst = min(sims)

    assert worst > 0.0, (
        f"DFC feedback is anti-gradient at init "
        f"[dendritic_effect='{dendritic_effect}']. "
        f"Q may not be correctly initialised to J^T.\n"
        f"  Per-layer cos_sims : {[f'{s:.4f}' for s in sims]}\n"
        f"  Worst layer        : {worst:.4f}"
    )


@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_dfc_derivative_flag_improves_gradient_alignment(tiny_network, tiny_batch, dendritic_effect):
    """
    Ablation hypothesis: use_derivative=True brings DFC closer to the true
    gradient by recovering the local f'(z_i) factor that pure DFC omits.

        Without: c_i =  Q_i u
        With:    c_i = (Q_i u) * f'(z_i)

    Tested across multiple random seeds to guard against a lucky single-seed
    result.  Two conditions must hold:
      1. Mean cosine similarity is higher with derivative than without,
         averaged across all seeds.
      2. The derivative flag improves alignment on a strict majority of seeds
         (>= 4 out of 5), allowing for one adversarial seed where the SiLU
         negative-derivative tail dominates.
    """
    seeds = [1, 2, 3, 7, 42]
    records = []  # (seed, mean_no_deriv, mean_with_deriv, sims_no, sims_with)

    for seed in seeds:
        torch.manual_seed(seed)
        x, _ = tiny_batch
        x = x.detach().requires_grad_(True)

        y = _prime_and_forward(tiny_network, x, dendritic_effect)
        u = torch.randn_like(y)

        # Both calls read pop.z; must happen before backward() releases the graph
        dfc_no_deriv   = tiny_network.DFC_project_feedback(u, use_derivative=False)
        dfc_with_deriv = tiny_network.DFC_project_feedback(u, use_derivative=True)
        autograd_grads = _autograd_grads(tiny_network, y, u)

        sims_no   = _cosine_sims(dfc_no_deriv,   autograd_grads)
        sims_with = _cosine_sims(dfc_with_deriv, autograd_grads)

        mean_no   = sum(sims_no)   / len(sims_no)
        mean_with = sum(sims_with) / len(sims_with)

        records.append((seed, mean_no, mean_with, sims_no, sims_with))

    # ── Aggregate ──────────────────────────────────────────────────────────
    overall_mean_no   = sum(r[1] for r in records) / len(records)
    overall_mean_with = sum(r[2] for r in records) / len(records)
    n_seeds_improved  = sum(1 for r in records if r[2] > r[1])

    # ── Assertions ─────────────────────────────────────────────────────────
    assert overall_mean_with > overall_mean_no, (
        f"use_derivative=True did NOT improve mean alignment across seeds "
        f"[dendritic_effect='{dendritic_effect}'].\n"
        f"  Overall mean without : {overall_mean_no:.4f}\n"
        f"  Overall mean with    : {overall_mean_with:.4f}"
    )

    min_seeds_required = len(seeds) - 1  # majority: 4 out of 5
    assert n_seeds_improved >= min_seeds_required, (
        f"use_derivative=True improved on only {n_seeds_improved}/{len(seeds)} seeds "
        f"[dendritic_effect='{dendritic_effect}']. "
        f"Required >= {min_seeds_required}.\n"
        f"  Per-seed means (no_deriv, with_deriv): "
        f"{[(f'{r[1]:.4f}', f'{r[2]:.4f}') for r in records]}"
    )

    # ── Diagnostic print (visible with pytest -s) ──────────────────────────
    print(f"\n[PASSED] derivative flag alignment improvement "
          f"[dendritic_effect='{dendritic_effect}']")
    print(f"  {'seed':<6} {'no_deriv':>10} {'with_deriv':>12} {'delta':>8}  per-layer-delta")
    for seed, mean_no, mean_with, sims_no, sims_with in records:
        delta = mean_with - mean_no
        layer_deltas = [f"{w-n:+.4f}" for w, n in zip(sims_with, sims_no)]
        improved = "✓" if mean_with > mean_no else "✗"
        print(f"  {seed:<6} {mean_no:>10.4f} {mean_with:>12.4f} {delta:>+8.4f}  {layer_deltas}  {improved}")
    print(f"  {'overall':<6} {overall_mean_no:>10.4f} {overall_mean_with:>12.4f} "
          f"{overall_mean_with - overall_mean_no:>+8.4f}  "
          f"({n_seeds_improved}/{len(seeds)} seeds improved)")

def test_invalid_mode_raises(tiny_network, tiny_batch):
    """Edge case check: Verifies validation code raises error on garbage strings."""
    x, y = tiny_batch
    controller = ControlMechanism(mode='nonsense')
    with pytest.raises(ValueError):
        controller.optimize_control_signal(x, y, tiny_network)

@pytest.mark.parametrize("mode", ["backprop", "pid"])
@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_weights_stay_unchanged(tiny_network, tiny_batch, mode, dendritic_effect):
    for pop in tiny_network.populations:
        pop.dendritic_effect = dendritic_effect

    x, y = tiny_batch 
    controller = ControlMechanism(mode=mode, max_steps=5)

    weights_before = [pop.W.weight.detach().clone() for pop in tiny_network.populations]
    controller.optimize_control_signal(x, y, tiny_network)
    weights_after = [pop.W.weight for pop in tiny_network.populations]

    for i, (W_before, W_after) in enumerate(zip(weights_before, weights_after)):
        assert torch.equal(W_before, W_after), (
            f"[{mode}] Layer {i} weights changed during control optimization."
        )
@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_initialized_controls_shape_and_magnitude(tiny_network, tiny_batch, mode): 
    x, y = tiny_batch
    controller = ControlMechanism(mode=mode, max_steps=5)
    expected = controller.initialize_controls(x.size(0), tiny_network.populations)
    c_star, _ = controller.optimize_control_signal(x, y, tiny_network)
    assert len(c_star) == len(expected)
    for actual, exp in zip(c_star, expected):
        assert actual.shape == exp.shape

@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_controls_have_false_required_grad(tiny_network, tiny_batch, mode): 
    x, y = tiny_batch
    controller = ControlMechanism(mode=mode, max_steps=5)
    c_star, _ = controller.optimize_control_signal(x, y, tiny_network)
    for c in c_star:
        assert not c.requires_grad, "Control signals should not require gradients."

@pytest.mark.parametrize("mode", ["backprop", "pid"])
@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_early_exit_at_convergence(tiny_network, tiny_batch, mode, dendritic_effect, huge_tolerance = 1000): 
    controller = controller = ControlMechanism(mode=mode)
    for pop in tiny_network.populations:
        pop.dendritic_effect = dendritic_effect

    x, y = tiny_batch
    controller.tolerance = huge_tolerance  # Set a huge tolerance to force early exit

    with torch.no_grad():
        y_baseline = tiny_network(x, control_signals=None, save_baseline=False)
    
    c_star, _ = controller.optimize_control_signal(x, y, tiny_network)

    # 1. No PID step actually executed → returned controls are still zeros.
    for i, c in enumerate(c_star):
        assert torch.allclose(c, torch.zeros_like(c)), (
            f"Layer {i}: expected zero c_star after early exit, "
            f"max abs value was {c.abs().max().item():.2e}"
        )

    # 2. `a_controlled` MUST be populated. Plasticity reads it; the dummy
    #    forward pass inside the `if step == 0:` branch is what guarantees this.
    for i, pop in enumerate(tiny_network.populations):
        assert pop.a_controlled is not None, (
            f"Layer {i}: a_controlled is None after early exit. "
            f"The dummy forward pass at step==0 didn't run — plasticity will crash."
        )

    # 3. With zero control signals, the controlled output should equal baseline.
    with torch.no_grad():
        y_controlled = tiny_network(x, control_signals=c_star, save_baseline=False)
    assert torch.allclose(y_controlled, y_baseline, atol=1e-6), (
        "With zero c_star, controlled output should match the no-control baseline."
    )

@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
@pytest.mark.parametrize(
    "mode, max_steps, min_relative_improvement",
    [
        # Backprop uses true gradients → expect aggressive convergence.
        ("backprop", 100, 0.02),
        # PID uses DFC-style random feedback → slower; expect modest but real progress.
        ("pid", 500, 0.005),
    ],
    ids=["backprop", "pid"],
)
def test_convergence_reduces_loss(
    tiny_network, tiny_batch, dendritic_effect, mode, max_steps, min_relative_improvement
):
    """
    The control optimiser should drive the network output toward the target.

    Both modes must:
      1. Take at least one optimization step.
      2. Reduce the loss below its baseline value.
      3. Achieve a mode-specific minimum relative improvement.

    Backprop uses autograd through the forward weights and should converge
    fast.  PID uses DFC's `Q_i u` feedback (Meulemans 2021) with random
    feedback matrices, which is gradient-aligned only in expectation, so
    we require a smaller improvement and grant more steps.
    """
    for pop in tiny_network.populations:
        pop.dendritic_effect = dendritic_effect
    controller = ControlMechanism(mode=mode, max_steps=max_steps)
    x, y = tiny_batch

    _, metrics = controller.optimize_control_signal(x, y, tiny_network)

    # ---- API-contract sanity --------------------------------------------
    assert metrics.initial_loss is not None, "initial_loss never recorded"
    assert metrics.final_loss is not None, "final_loss never recorded"
    assert metrics.steps_taken > 0, "optimizer never took a step"
    assert len(metrics.loss_history) >= 2, "loss_history should contain at least baseline + one update"

    # ---- Main behavioural check -----------------------------------------
    history = metrics.loss_history

    if mode == "backprop":
        monotone = all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
        assert monotone, _failure_report(
            mode, metrics, reason="Backprop loss history is not monotonically decreasing"
        )

    rel_improvement = metrics.improvement / metrics.initial_loss
    assert rel_improvement >= min_relative_improvement, _failure_report(
        mode, metrics,
        reason=(
            f"improvement {rel_improvement*100:.2f}% "
            f"< required {min_relative_improvement*100:.2f}%"
        ),
    )


def _failure_report(mode: str, metrics, reason: str) -> str:
    """Verbose error message """
    history = metrics.loss_history
    first = history[:5]
    last = history[-5:]
    monotone = all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
    return (
        f"\n[mode={mode}] {reason}\n"
        f"  initial_loss : {metrics.initial_loss:.6f}\n"
        f"  final_loss   : {metrics.final_loss:.6f}\n"
        f"  improvement  : {metrics.improvement:.6e}\n"
        f"  steps_taken  : {metrics.steps_taken}\n"
        f"  converged    : {metrics.converged}\n"
        f"  monotone↓    : {monotone}\n"
        f"  history[:5]  : {[f'{v:.4f}' for v in first]}\n"
        f"  history[-5:] : {[f'{v:.4f}' for v in last]}"
    )

@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_pid_does_not_diverge(tiny_batch, dendritic_effect):
    x, y = tiny_batch
    diverged_seeds = []

    for seed in range(5):
        set_all_seeds(seed)
        net = Network(pop_sizes=[2, 4, 1])
        for pop in net.populations:
            pop.dendritic_effect = dendritic_effect
            
        controller = ControlMechanism(mode='pid', max_steps=100)
        _, metrics = controller.optimize_control_signal(x, y, net)

        history = metrics.loss_history
        increases = sum(b > a + 1e-9 for a, b in zip(history, history[1:]))
        total_steps = len(history) - 1
        if increases / total_steps > 0.6:
            diverged_seeds.append(seed)

    assert not diverged_seeds, (
        f"[{dendritic_effect}] PID diverged on seeds: {diverged_seeds}"
    )




    







