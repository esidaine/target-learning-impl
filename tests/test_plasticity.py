from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from _helpers import _diagnose_layerwise_mismatches
import torch 
import torch.nn.functional as F
import pytest

@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_zero_error_produces_zero_update(tiny_network, tiny_batch, mode):
    """If target == baseline_pred, nothing should change."""
    x, _ = tiny_batch   
    controller = ControlMechanism(mode=mode, lr_c=0.1, max_steps=60)
    plasticity = Plasticity(lr_theta=0.2)

    # Run baseline forward pass to get baseline predictions
    with torch.no_grad():
            baseline_pred = tiny_network(x, control_signals=None, save_baseline=True)

    # Set error to zero by making target equal to baseline_pred - there is nothing to learn
    target_y = baseline_pred.clone()

    # Snapshot the weights so we can check they don't move
    weights_before = [pop.W.weight.detach().clone() for pop in tiny_network.populations]

    batch_size = x.size(0)
    initial_controls = controller.initialize_controls(batch_size, tiny_network.populations)
    # Get controls - this should give controls that are zero, since the error is zero
    if mode == 'backprop':
        controller._optimize_via_backprop(initial_controls, x, target_y, tiny_network)
    elif mode == 'pid':
        controller._optimize_via_pid(initial_controls, x, target_y, tiny_network, baseline_pred)
    else:
        raise ValueError("Mode must be 'backprop' or 'pid'")

    # Update weigths - this should do nothing since controls are zero
    plasticity.update_weights(tiny_network, x)

    # Assertions
    weights_after = [pop.W.weight.detach().clone() for pop in tiny_network.populations]

    # Compare weights before and after - they should be the same
    _diagnose_layerwise_mismatches(
        pairs=zip(weights_before, weights_after),
        tol=1e-10,
        label="Weight change after zero-error training step",
    )

    # Compare controls - they should be zero
    _diagnose_layerwise_mismatches(
        pairs=[(pop.a_controlled, pop.a_baseline) for pop in tiny_network.populations],
        tol=1e-10,
        label="a_controlled vs a_baseline at zero error",
    )
    

@pytest.mark.parametrize(
    "network_fixture,batch_fixture",
    [
        ("tiny_network", "tiny_batch"),                # XOR: 1-D output
        ("mnist_network", "mnist_tiny_batch"),    # MNIST-shape: 10-D output
    ],
)
@pytest.mark.parametrize("mode", ["backprop", "pid"])
def test_dfc_update_aligns_with_gn_at_small_beta(
    network_fixture, batch_fixture, mode, request
):
    """As beta -> 0, the DFC weight update should align with the
    Gauss-Newton (GN) update direction.

    Theoretical Background (Meulemans et al., NeurIPS 2021, Thm 2):
    ---------------------------------------------------------------
    Under stable dynamics, DFC naturally performs second-order Gauss-Newton (GN) 
    optimization rather than first-order gradient descent (BP). 
    - BP calculates steepest descent: delta_W ∝ J^T * δ
    - DFC (GN) accounts for curvature: delta_W ∝ J^T * (J * J^T)^(-1) * δ
    
    The PID controller's feedback loop naturally computes the inverse of the forward 
    network sensitivities. It finds the path of least resistance through the network's 
    non-linearities, mathematically represented by the Moore-Penrose pseudo-inverse 
    term (J * J^T)^(-1).

    The 1-D Output Exception (Why this test works):
    -----------------------------------------------
    In a multi-dimensional output network (e.g., 10 classes), (J * J^T)^(-1) is a 
    matrix that rotates the error vector. In that scenario, DFC and BP point in 
    fundamentally different directions, and cosine similarity would be low. 
    
    However, if the network output is 1-D (a single scalar), (J * J^T) evaluates to 
    a 1x1 scalar. Because multiplying by a scalar can only scale a vector and cannot 
    rotate it, the GN update and the BP update point in the exact same direction.

    This test compares the DFC update to the GN reference, which is the
    correct comparison in both cases:
      - for the 1-D XOR network, GN equals BP-gradient direction, so the
        test also implicitly checks alignment with BP;
      - for the 10-D MNIST-shape network, GN differs from BP by an
        (output_dim x output_dim) rotation, and only GN is the right
        reference.

    For each beta in a decreasing sweep we check two things:
      (1) cosine similarity is monotonically non-decreasing as beta shrinks
          (the higher-order Taylor terms vanish, the linear limit emerges);
      (2) at the smallest beta, cosine similarity is tight (>= 0.95).
    """
    network = request.getfixturevalue(network_fixture)
    x, _ = request.getfixturevalue(batch_fixture)

    # max_steps generous so PID has time to settle, especially for the
    # 10-D MNIST fixture where the controller has more dimensions to balance.
    controller = ControlMechanism(mode=mode, lr_c=0.05, max_steps=300)
    plasticity = Plasticity(lr_theta=0.1)

    # Run baseline forward pass to get baseline predictions
    with torch.no_grad():
        baseline_pred = network(x, control_signals=None, save_baseline=True)
 
    weights_before = [pop.W.weight.detach().clone() for pop in network.populations]

    # Fix the nudge direction, only sweep its magnitude (beta) 
    nudge_direction = torch.randn_like(baseline_pred) # in output space
    nudge_direction = nudge_direction / nudge_direction.norm()

    nudge_scaling_factors = [1.0, 0.1, 0.01, 0.001] # # these are the β values from the DFC Taylor expansion
    cos_sims = []

    for magnitude in nudge_scaling_factors:
        # Restore weights and reset dynamic state
        for pop, w in zip(network.populations, weights_before):
            pop.W.weight.data.copy_(w)
            pop.repolarize()

        # Construct an artifical taget that sits a small distance away from the current output
        # The direction is fixed, but the magnitude is changing to sweep beta -> 0
        # We construct the target so that it lies along the random nudge direction
        target_y = baseline_pred + magnitude * nudge_direction 

        # Find optimal controls in the chosen mode.
        initial_controls = controller.initialize_controls(x.size(0), network.populations)
        if mode == "backprop":
            controller._optimize_via_backprop(initial_controls, x, target_y, network)
        else:
            controller._optimize_via_pid(
                initial_controls, x, target_y, network, baseline_pred
            ) 

        # Compute the weight updates (this is the update we are testing)
        # Note that we compute them, but do not update the weights
        delta_W_flat = _compute_weight_updates_flat(network, plasticity, x)

        # Compute the reference Gauss-Newton update direction (this is the gold standard we are testing against)
        # TODO
        # filler for gn_flat to avoid errors
        gn_flat = _compute_gn_update_flat(network, x, target_y, baseline_pred)

        # Compare the angles between them
        # Gives a single number between -1 and 1, where 1 means perfectly aligned, 0 means orthogonal, and -1 means opposite directions 
        cos = F.cosine_similarity(
            delta_W_flat.unsqueeze(0), gn_flat.unsqueeze(0), dim=1
        ).item()
        cos_sims.append(cos)
        
    print(f"\n[{network_fixture} / mode={mode}] cos(DFC, GN) sweep:")
    for b, c in zip(nudge_scaling_factors, cos_sims):
        print(f"  beta = {b:7.4g}    cos = {c:.4f}")

    # Assertions
    # TODO
        


def _compute_weight_updates_flat(network, plasticity, x):
    """Apply the learning rule per layer (without writing to W) and
    flatten the resulting weight updates into one long vector."""
    parts = []
    for i, pop in enumerate(network.populations):
        a_pre = x if i == 0 else network.populations[i - 1].a_controlled

        delta_W_i = plasticity.learning_rule(a_pre, pop.a_baseline, pop.a_controlled)

        parts.append(delta_W_i.reshape(-1))

    return torch.cat(parts).detach()

def _compute_gn_update_flat(network, x, target_y, baseline_pred, gamma=1e-6):
    """Compute the Gauss-Newton update direction, flattened across all layers.

    Returns:  J^T (J J^T + gamma * I)^(-1) (target_y - baseline_pred)

    where J is the per-sample Jacobian of the network output w.r.t. all
    weights, stacked over the batch (so J has shape [B*D_out, total_params]).
        - (target_y - baseline_pred) is the error vector in output space
        - (JJ^T), is an approximation of the Hessian matrix, which accounts for the curvature of the loss landscape
        - (J*J^T)^(-1) compensates for the varying the sensitivities and interferences
            between different output dimensions, effectively scaling and rotating the error vector
        - J^T (J J^T)^(-1) is the Moore-Penrose Right Pseudo-Inverse of J, 
            which takes an error in output space and calculates the step in weight space
        - linear algebra guarantees that you will find the Least-Norm Solution.

    From Theorem 2 of Meulemans et al. (Deep Feedback Control, NeurIPS 2021):
    in the steady-state, small-error limit, the DFC weight update should align with the GN update. 
    Should pass for both the 1-D XOR network and the 10-D MNIST-shaped network
    """
    weights = [pop.W.weight for pop in network.populations]
    original_flags = [w.requires_grad for w in weights]
    for w in weights:
        w.requires_grad_(True)

    try: 
        # build jacobian J 
        # build gn_flat
        gn_flat = torch.zeros(sum(w.numel() for w in weights))
        ...
    
    finally: 
        # restore original requires_grad flags so we don't mess up other tests
        for w, flag in zip(weights, original_flags):
            w.requires_grad_(flag)
        ...

    return gn_flat.detach()


