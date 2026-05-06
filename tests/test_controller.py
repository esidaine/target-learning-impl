import torch
from core.controllers import ControlMechanism
from _helpers import _diagnose_layerwise_mismatches
from torch.nn import functional as F

def test_get_local_controls_matches_autograd(tiny_network, tiny_batch):
    """
    Tests that the pid controls computed by get_local_controls match the autograd-computed controls, for a single forward pass with zero controls.
    """
    # Set up tiny network and batch
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
    cos_sim = F.cosine_similarity(pid_controls[0].flatten(), autograd_controls[0].flatten(), dim=0)
    assert cos_sim > 0.99, "Cos similarity: PID direction diverges from Autograd gradient."

    _diagnose_layerwise_mismatches(
        pairs=zip(pid_controls, autograd_controls),
        tol=1e-5,
        label="pid vs autograd controls",
    )


