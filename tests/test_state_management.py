import torch
from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from core.trainer import Trainer

def test_repolarize_clears_physical_state(tiny_network, tiny_batch):
    """
    Ensures that when a new batch arrives (save_baseline=True), all physical state 
    from the previous batch's control phase is completely wiped, preventing state bleeding.
    """
    x, _ = tiny_batch

    dummy_controls = [
        torch.full((x.size(0), pop.num_neurons), 0.5, device=x.device)
        for pop in tiny_network.populations
    ]

    # 1. Simulate first batch 
    # Baseline pass to initialize a_baseline and a_controlled for the first batch
    tiny_network(x, control_signals=None, save_baseline=True)
    # Control signals pass to update a_controlled for the first batch
    tiny_network(x, control_signals=dummy_controls, save_baseline=False, dynamic_step=True)

    # 2. Simulate the arrival of Batch 2 (save_baseline=True triggers repolarize)
    tiny_network(x, control_signals=None, save_baseline=True)

    # 3. Assert state is completely cleared
    for i, pop in enumerate(tiny_network.populations):
        assert pop.a_controlled is None, (
            f"STATE LEAK: Layer {i}'s a_controlled was not reset to None "
            f"when the new batch arrived."
        )


def test_no_computational_graph_leakage(tiny_network, tiny_batch):
    """
    Ensures that the custom Target Learning manual weight updates do not accidentally 
    accumulate PyTorch Autograd gradients, which would cause memory leaks over epochs.
    """
    controller = ControlMechanism(mode='backprop', lr_c=0.1, max_steps=5)
    plasticity = Plasticity(lr_theta=0.2)
    trainer = Trainer(tiny_network, controller, plasticity)
    
    x, _ = tiny_batch
    out_dim = tiny_network.populations[-1].num_neurons
    random_target = torch.randn(x.size(0), out_dim)

    # 1. Run one complete training loop
    trainer.controller.optimize_control_signal(x, random_target, trainer.network)
    plasticity.update_weights(trainer.network, x)

    # 2. Verify Weights have NO autograd gradients
    for i, pop in enumerate(tiny_network.populations):
        assert pop.W.weight.grad is None, (
            f"GRAPH LEAK: Layer {i}'s weights accumulated PyTorch gradients! "
            f"Plasticity should be manual. Check for missing @torch.no_grad() decorators."
        )

    # 3. Verify baseline and controlled states are detached from the graph
    for i, pop in enumerate(tiny_network.populations):
        # We only check if they exist (they might be None)
        if pop.a_baseline is not None:
            assert not pop.a_baseline.requires_grad, f"Layer {i} a_baseline requires grad."
            
        # In 'backprop' mode, a_controlled might temporarily require grad during optimization,
        # but once optimization is done and we move on, it shouldn't be holding a graph 
        # that affects the parameters. Since your clean-up loops detach a_controlled:
        if pop.a_controlled is not None:
            assert not pop.a_controlled.requires_grad, (
                f"MEMORY LEAK: Layer {i} a_controlled is still attached to the computational "
                f"graph after the control phase finished. This will blow up your VRAM."
            )