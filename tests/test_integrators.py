import torch
from typing import Optional

def test_firing_rate_dynamics_converge(tiny_network, tiny_batch): 
    """
    This test checks class FiringRateDynamics. 
    a_controlled chases target_activation over time. Convergence means a_controlled has reached its fixed point
    for the currently held control signals. The fixed point is just a_controlled = target_activation
    """
    x,_ = tiny_batch

    # 1. Free pass: seeds a_baseline in every population so a_controlled
    #    has something to initialize from on the first dynamic step.
    tiny_network(x, control_signals=None, save_baseline=True)

    # NON-ZERO controls so we actually force the activations to change
    dummy_controls = [
        torch.full((x.size(0), pop.num_neurons), 0.5, device=x.device)
        for pop in tiny_network.populations
    ]

    max_time_steps = 200  # A cutoff so we don't loop forever if it's unstable
    network_converged: Optional[int] = None 
    previous_activation: Optional[torch.Tensor] = None
    next_activations: Optional[torch.Tensor] = None

    for step in range(max_time_steps):
        next_activations = tiny_network(x, control_signals=dummy_controls, save_baseline=False, dynamic_step=True)
        
        # Check for each pop whether firings have settled
        # returns True only when every single one currently reports firing_settled == True
        if all(pop.firing_settled for pop in tiny_network.populations):
            network_converged = step
            break
        
        assert next_activations is not None
        previous_activation = next_activations.detach().clone()
            
        

    # Check converged within reasonable time
    assert network_converged is not None, (
    f"Activations did not settle within {max_time_steps} steps"
    )

    # At the fixed point, every layer's a_controlled should equal its own target.
    for i, pop in enumerate(tiny_network.populations):
        assert pop.a_controlled is not None, f"pop[{i}].a_controlled was never set"
        assert pop.target_activation is not None, f"pop[{i}].target_activation was never set"
        assert torch.allclose(
            pop.a_controlled, pop.target_activation,  atol=1e-3
        ), (
            f"Layer {i}: a_controlled does not match target_activation. "
            f"max abs diff = "
            f"{(pop.a_controlled - pop.target_activations).abs().max().item():.6f}"
        )

    # No further movement at the fixed point.
    if previous_activation is not None and next_activations is not None:
        assert torch.allclose(next_activations, previous_activation, atol=1e-3), (
            "Output is still moving at the reported convergence step"
        )

    # Check wether it actually moved 
    for pop in tiny_network.populations:
        assert not torch.allclose(pop.a_controlled, pop.a_baseline, atol=1e-3), \
            "Firing rate dynamics never actually moved from baseline."


    
    







