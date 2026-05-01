import torch
from models.network import Network
from core.controllers import ControlMechanism

def test_get_local_controls_matches_autograd(tiny_network, tiny_batch):
    # Set up tiny detwork and batch
    torch.manual_seed(0)
    x, _ = tiny_batch
    x = x.detach().requires_grad_(True)   # put x into the autograd graph so that we can backprop through it
    controller = ControlMechanism(mode='backprop')

    initial_controls = controller.initialize_controls(
        batch_size = x.size(0),
        neuron_populations = tiny_network.populations
    ) # already returns zeros with requires_grad=True

    # ONE forward pass: zero controls with requires_grad=True.
    # This populates pop.z AND puts it in the autograd graph.
    y = tiny_network.forward(x, control_signals=initial_controls, save_baseline=False)

    # Pick an arbitrary global control signal 
    global_control = torch.randn_like(y)

    # Pass global control through the network 
    pid_controls = tiny_network.get_local_controls(global_control)

    # Get the autograd computed controls by backprop
    for pop in tiny_network.populations:
        # pop.z is 
        # Save the gradient on this intermediate tensor when backward runs
        pop.z.retain_grad()

    # Compute derivatives, pretending that the gradient at the output is global_control
    (global_control * y).sum().backward()

    # Read out the saved gradients at each layer's z.
    autograd_controls = [pop.z.grad.clone() for pop in tiny_network.populations]

    # Compare the two sets of controls
    _diagnose_mismatches(pid_controls, autograd_controls)


def _diagnose_mismatches(manual, autograd, rtol=1e-5, atol=1e-6):
    """ Summary across all layers."""
    rows = []
    # Check each layer's controls for closeness, and if not, gather stats on the mismatch
    for i, (m, a) in enumerate(zip(manual, autograd)):
        if torch.allclose(m, a, rtol=rtol, atol=atol):
            continue

        # Get mismatch
        diff = (m - a).abs()

        # Calculate the max and mean difference to understand the scale of the mismatch
        max_d, mean_d = diff.max().item(), diff.mean().item()

        # Identify whetehr diff values cluster tightly around the mean
        kind = "systematic" if max_d < 3 * mean_d else "outlier"
        rows.append(f"  L{i}: mean={mean_d:.2e}  max={max_d:.2e}  ({kind})")

    if rows:
        raise AssertionError(
            f"\n{len(rows)}/{len(manual)} layers mismatch:\n" + "\n".join(rows)
        )