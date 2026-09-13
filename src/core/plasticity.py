import torch
from utils.utils import get_logger

logger = get_logger()


class Plasticity:
    """
    Updates the weights based on the learning rule

    num_neurons is the number of neurons in one specific layer (or "population")
    """

    def __init__(self, lr_w: float):
        """Initialize the weight-update rules used during the plasticity phase.

        Args:
            lr_w (float): Learning rate applied to the weight and bias increments.

        Returns:
            None.
        """
        self.lr_w = lr_w  # Learning rate for the network weights

    def learning_rule(self, a_pre, a_baseline, a_controlled):
        """Compute the local weight and bias updates from the activation difference.

        Args:
            a_pre (torch.Tensor): Presynaptic activations feeding the current population.
            a_baseline (torch.Tensor): Baseline activation before control is applied.
            a_controlled (torch.Tensor): Activation after control has been applied.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The matrix update for the weights and the
                vector update for the biases.
        """
        # 1. Calculate the difference between the target activation and baseline activation
        # Shape: [batch_size, num_neurons]
        activation_delta = a_controlled - a_baseline

        # 2. Matrix multiplication to get the outer product, summing across the batch
        # activation_delta.T shape: [num_neurons, batch_size]
        # a_pre shape: [batch_size, num_inputs]
        # delta_W shape: [num_neurons, num_inputs]
        delta_W = torch.matmul(activation_delta.T, a_pre)

        # 3. Divide by batch size to get the average weight update to compute the average weight update per training example
        batch_size = a_pre.size(0)
        delta_W_batched = delta_W / batch_size

        # 4. Calculate bias updates as the average error across the batch
        delta_b = activation_delta.mean(dim=0)
        return delta_W_batched, delta_b

    @torch.no_grad()  # Turn off gradients since we are doing manual weight updates
    def update_weights(self, network, sensory_inputs):
        """
        Applies the learning rule to every population in the network.
        This must be called AFTER the ControlMechanism has found c* and
        populated a_baseline and a_controlled.

        Args:
            network (Network): Network whose forward weights are updated.
            sensory_inputs (torch.Tensor): Current batch input used by layer 0.

        Returns:
            None.
        """

        # Iterate over all populations/layers in the network
        for i, pop in enumerate(network.populations):
            # 1. Determine the presynaptic input for this specific population

            if i == 0:
                # For the first layer, the presynaptic inputs are the raw sensory data (images)
                a_pre = sensory_inputs
            else:
                # For deeper layers, the presynaptic inputs are the CONTROLLED activations from the previous layer
                a_pre = network.populations[i - 1].a_controlled

            # 2. Grab the saved postsynaptic states for this population
            a_base = pop.a_baseline
            a_ctrl = pop.a_controlled

            # 3. Calculate the weight update matrix
            delta_W, delta_b = self.learning_rule(a_pre, a_base, a_ctrl)

            # 4. Apply the update to the population's weights
            pop.W.weight.add_(delta_W, alpha=self.lr_w)

            # Also update the bias term
            pop.W.bias.add_(delta_b, alpha=self.lr_w)
