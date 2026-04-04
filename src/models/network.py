import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath('..'))
from src.utils.utils import get_logger

logger = get_logger()


class Network(nn.Module):
    def __init__(self, pop_sizes):
        super().__init__()
        
        # Note that pop[0] referrs to the first hidden layer, not the input layer
        self.populations = nn.ModuleList([
            NeuralPopulation(pop_sizes[i], pop_sizes[i+1])
            for i in range(len(pop_sizes) - 1)
        ])

    def forward(self, sensory_inputs, control_signals=None, save_baseline=False):
        """
        Passes data through the network.
        - If control_signals is None, it acts as the Baseline Pass (c_n = 0).
        - If save_baseline=True, it locks in the baseline states.
        """
        # The firing rate based on the sensory inputs
        activation = sensory_inputs 

        for i, pop in enumerate(self.populations):
            # 1. Grab the control signal, or default to zeros for baseline
            if control_signals is not None:
                c_n = control_signals[i]
            else:
                # Dummy variable to pass to dendritic_proc(c_n) to calculate q_c
                c_n = torch.zeros(activation.size(0), int(pop.num_neurons), device=pop.W.weight.device)

            # 2. Calculate the firing rate given the sensory inputs and control signal
            activation = pop.firing_rate(activation, c_n)

            # 3. Save activation states 
            if save_baseline:
                # .detach().clone() creates a static copy completely disconnected from PyTorch's autograd engine.
                pop.a_baseline = activation.detach().clone()

            elif control_signals is not None:
                # During the settling phase, we want to keep PyTorch’s Autograd graph attached so we can backpropagate errors to c_n.
                # Allows PyTorch to trace the errors back to cn​ during the .backward() step.
                pop.a_controlled = activation

        return activation

class NeuralPopulation(nn.Module):
    def __init__(self, num_inputs, num_neurons):
        """
        Represents a group of neurons.
        Builds the anatomy and processing rules for a single neuron, which can be replicated across the population.
        """
        super().__init__()
        self.num_neurons = num_neurons
        
        # 1. HOLDING WEIGHTS, disable biases and gradient tracking
        self.W = nn.Linear(num_inputs, num_neurons, bias=False)
        self.W.weight.requires_grad = False
        
        # 2. INITIALIZING STATE MEMORY 
        self.a_baseline = None # Activation for the first guess
        self.a_controlled = None # The dynamic physical state that changes over time given the control signal (top-down input)
    
    def dendritic_proc(self, c_n):
        return torch.sigmoid(c_n) - 0.5  # Range is now [-0.5, 0.5]

    def bottom_up_proc(self, sensory_inputs):
        """
        Gets the sensory inputs and sums them up (z_n). The total bottom-up input for our current neuron (neuron n) is 
        the activation of a neuron from the previous layer (neuron m) times their connecting weight (wmn​), summed up across 
        all the neurons in that previous layer. Then applies leaky ReLU activation function. 
        """
        # Multiply the inputs by the weights, identical to: z = np.dot(sensory_inputs, weights) 
        z = self.W(sensory_inputs) 
        # Apply leaky relu function
        return F.leaky_relu(z, negative_slope=0.01)

    def firing_rate(self, sensory_inputs, c_n, beta=1.0):
        """
        Calculates the final firing rate of the neuron by combining the bottom-up 
        drive with the top-down dendritic modulation.
        """
        # 1. Get the bottom-up activation
        phi_z = self.bottom_up_proc(sensory_inputs)
        
        # 2. Get the top-down apical activation
        q_c = self.dendritic_proc(c_n)
        
        # 3. Combine them with multiplicative effect (element-wise multiplication)
        r = (beta * q_c + 1) * phi_z 
        
        return r
    
    