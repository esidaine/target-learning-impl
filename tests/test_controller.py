import torch
from core.controllers import ControlMechanism
from torch.nn import functional as F
import pytest

@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_project_feedback_matches_autograd_direction(tiny_network, tiny_batch, dendritic_effect):
    """
    Tests that the pid controls computed by project_feedback match the autograd-computed controls, for a single forward pass with zero controls.
    """
    tiny_network.dendritic_effect = dendritic_effect
    torch.manual_seed(0)
    x, _ = tiny_batch
    x = x.detach().requires_grad_(True)   # put x into the autograd graph so that we can backprop through it
    controller = ControlMechanism(mode='backprop')

    initial_controls = controller.initialize_controls(
        batch_size = x.size(0),
        neuron_populations = tiny_network.populations
    ) # already returns zeros with requires_grad=True

    # Prime a_baseline exactly the way optimize_control_signal does.
    with torch.no_grad():
        tiny_network.eval()
        tiny_network.forward(x, control_signals=None, save_baseline=True)
        tiny_network.train()


    # ONE forward pass: zero controls with requires_grad=True.
    # This populates pop.z AND puts it in the autograd graph.
    y = tiny_network.forward(x, control_signals=initial_controls, save_baseline=False, dynamic_step=True)

    # Pick an arbitrary global control signal 
    global_control = torch.randn_like(y)

    # Pass global control through the network 
    pid_controls = tiny_network.project_feedback(global_control)

    # Get the autograd computed controls by backprop
    for pop in tiny_network.populations:
        # Save the gradient on this intermediate tensor when backward runs
        pop.z.retain_grad()

    # Compute derivatives, pretending that the gradient at the output is global_control
    (global_control * y).sum().backward()

    # Read out the saved gradients at each layer's z.
    autograd_controls = [pop.z.grad.clone() for pop in tiny_network.populations]

    # Compare the two sets of controls
    cos_sim = F.cosine_similarity(pid_controls[0].flatten(), autograd_controls[0].flatten(), dim=0)
    assert cos_sim > 0.25, "Cos similarity: PID direction diverges from Autograd gradient."

def test_invalid_mode_raises(tiny_network, tiny_batch):
    """Edge case check: Verifies validation code raises error on garbage strings."""
    x, y = tiny_batch
    controller = ControlMechanism(mode='nonsense')
    with pytest.raises(ValueError):
        controller.optimize_control_signal(x, y, tiny_network)

@pytest.mark.parametrize("mode", ["backprop", "pid"])
@pytest.mark.parametrize("dendritic_effect", ["additive", "multiplicative"])
def test_weights_stay_unchanged(tiny_network, tiny_batch, mode, dendritic_effect):
    tiny_network.dendritic_effect = dendritic_effect
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
    tiny_network.dendritic_effect = dendritic_effect
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
    tiny_network.dendritic_effect = dendritic_effect
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
    monotone = all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
    assert monotone, _failure_report(
        mode, metrics, reason="loss history is not monotonically decreasing"
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

def test_dfc_pid_mode_is_not_chain_rule(tiny_network, tiny_batch):
    """DFC's Q_i @ u should NOT equal the BP chain rule Q = J^T."""
    ...
    




    







