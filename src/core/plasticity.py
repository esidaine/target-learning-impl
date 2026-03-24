import torch
import os
import sys
sys.path.append(os.path.abspath('..'))
from utils import get_logger

logger = get_logger()

class Plasticity:
    """
    Updates the weights based on the learning rule

    num_neurons is the number of neurons in one specific layer (or "population")
    """

    def __init__(self, lr_theta=0.01):
        self.lr_theta = lr_theta # Learning rate for the network weights

    def learning_rule(self, a_pre, a_baseline, a_controlled):  
        # 1. Calculate the difference between the target state and baseline state
        # Shape: [batch_size, num_neurons]
        errors = a_controlled - a_baseline 
        
        # 2. Matrix multiplication to get the outer product, summing across the batch
        # errors.T shape: [num_neurons, batch_size]
        # a_pre shape: [batch_size, num_inputs]
        # delta_W shape: [num_neurons, num_inputs]
        delta_W = torch.matmul(errors.T, a_pre)

        # 3. Divide by batch size to get the average weight update
        batch_size = a_pre.size(0)
        return delta_W / batch_size

    @torch.no_grad() # Turn off gradients since we are doing manual weight updates
    def train_single_step(self, network, sensory_inputs): 
        """
        Applies the learning rule to every population in the network.
        This must be called AFTER the ControlMechanism has found c* and 
        populated a_baseline and a_controlled.
        """

        # Iterate over all populations/layers in the network
        for i, pop in enumerate(network.populations):
            # 1. Determine the presynaptic input (a_m) for this specific population
            if i == 0:
                # For the first layer, the presynaptic inputs are the raw sensory data (images)
                a_pre = sensory_inputs
            else:
                # For deeper layers, the presynaptic inputs are the baseline activations from the previous layer
                a_pre = network.populations[i-1].a_baseline
                
            # 2. Grab the saved postsynaptic states (a_n) for this population
            a_base = pop.a_baseline
            a_ctrl = pop.a_controlled
            
            # 3. Calculate the weight update matrix
            delta_W = self.learning_rule(a_pre, a_base, a_ctrl)
            
            # 4. Apply the update directly to the population's weights
            # We use += because we want to push the baseline state TOWARDS the controlled target state
            pop.W.weight.data += self.lr_theta * delta_W