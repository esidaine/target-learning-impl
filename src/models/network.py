import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath('..'))
from src.utils.utils import get_logger
from typing import Optional
from src.core.euler_integrators import FiringRateDynamics

logger = get_logger()


class Network(nn.Module):
    def __init__(self, pop_sizes):
        super().__init__()
        
        # Note that pop[0] referrs to the first hidden layer, not the input layer
        self.populations: nn.ModuleList = nn.ModuleList([
            NeuralPopulation(pop_sizes[i], pop_sizes[i+1])
            for i in range(len(pop_sizes) - 1)
        ])

    def get_local_controls(self, global_control):
            """
            Computes local control signals (c_n) so that the network produces the correct output. By routing the 
            signal backward through the network using the Chain Rule. Note that the weights are frozen. 

            Finds the target state (Psi_i) for the neurons in a given hidden layer 'i'

            The universal equation executed across the layers is:
                Psi_i = (Psi_{i+1} @ W_{i+1}^T) * f'(z_i)

            Where:
            - '@' represents matrix multiplication (dot product).
            - 'W^T' is the transpose of the forward weight matrix.
            - '*' represents element-wise multiplication.
            """
            # Counts the number of hidden layers to know how deep the network is
            L = len(self.populations) 

            # Creating a fixed array of slots for storing the target controls, slot with idx 0 corresponds to the first hidden layer etc. 
            control_targets = [None] * L

            # Pre-compute f'(z), the activation derivatives for all neurons per layers/populations
            neuron_sensitivities = []
            for pop in self.populations:
                assert isinstance(pop, NeuralPopulation) # Type guard
                neuron_sensitivities.append(pop.get_bottom_up_activation_derivatives())

            backward_signal = global_control # This is the initial signal that we want to propagate backward

            # Traverse backwards from the output layer to the first hidden layer
            for i in reversed(range(L)):
                # Get the sensitivities f' for the current layer/ population
                layer_sensitivity = neuron_sensitivities[i]

                if i == L - 1:
                   layer_target_controls = backward_signal * layer_sensitivity

                else:
                    # ! WILL BE CHANGED TO ALIGN WITH DFC 
                    back_weights = self.populations[i+1].W.weight # This is already the transpose
                    # We take the message from the layer above and push it backward through the feedback wiring
                    # We multiply the signal by the neuron's sensitivity
                    layer_target_controls = torch.matmul(backward_signal, back_weights) * layer_sensitivity
                
                # Store the target control for this layer	
                control_targets[i] = layer_target_controls

                # Update the backward signal for the next iteration (the next layer down)
                backward_signal = layer_target_controls 
       
            return control_targets	



    def forward(self, sensory_inputs, control_signals=None, save_baseline=False, dynamic_step=False):
        """
        Passes data through the network.
        - If control_signals is None, it acts as the Baseline Pass (c_i = 0).
        - If save_baseline=True, it locks in the baseline states.
        """
        # The firing rate based on the sensory inputs
        pop_activations = sensory_inputs 

        for i, pop in enumerate(self.populations):
            assert isinstance(pop, NeuralPopulation)

            # 1. Grab the control signal for the i-th population, or default to zeros for baseline
            if control_signals is not None:
                pop_controls = control_signals[i]
            else:
                # Dummy variable to pass to dendritic_proc(c) to calculate q_c
                pop_controls = torch.zeros(pop_activations.size(0), int(pop.num_neurons), device=pop.W.weight.device)

            # 2. Calculate the firing rate for a given layer given the sensory inputs and control signal
            pop_activations = pop.firing_rate(pop_activations, pop_controls, dynamic_step=dynamic_step)

            # 3. FREE PHASE: Save the baseline activations
            # A new batch has arrived, so reset all dynamic memory from the previous batch.
            # Only save exactly once per batch, before any control signals have been applied. 
            if save_baseline:
                # .detach().clone() creates a static copy completely disconnected from PyTorch's autograd engine.
                pop.a_baseline = pop_activations.detach().clone()
                # Reset dynamic memory for safety
                pop.repolarize()

            # 3. CONTROL PHASE: Update the activation state given the nudge from the control signal 
            elif control_signals is not None:
                # For backprop mode: Keep PyTorch’s Autograd graph attached so we can backpropagate errors to c
                # Note that we do not detach here, we want to keep the graph for backpropagation
                pop.a_controlled = pop_activations

        return pop_activations
    

class NeuralPopulation(nn.Module):
    def __init__(self, num_inputs, num_neurons, leaky_slope=0.01):
        """
        Represents a group of Multi-Compartment neurons with specific anatomy and processing rules of top-down and bottom-up input.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.leaky_slope = leaky_slope
        
        # 1. HOLDING WEIGHTS, disable biases and gradient tracking
        self.W = nn.Linear(num_inputs, num_neurons, bias=False)
        self.W.weight.requires_grad = False

        self.z: Optional[torch.Tensor] = None # Bottom Up 
        
        # 2. INITIALIZING STATE MEMORY 
        self.a_baseline: Optional[torch.Tensor] = None # Activation for the first guess
        self.a_controlled: Optional[torch.Tensor] = None # The dynamic physical state that changes over time given the control signal (top-down input)

        # 3. Internal dynamics for the firing rate to evolve over time
        self.dynamics = FiringRateDynamics(dt=0.1, tau=1.0)
    
    def dendritic_proc(self, signal):
        return torch.tanh(signal) + 1

    def bottom_up_proc(self, sensory_inputs):
        """
        Gets the sensory inputs and sums them up (z_n). The total bottom-up input for our current neuron (neuron n) is 
        the activation of a neuron from the previous layer (neuron m) times their connecting weight (wmn​), summed up across 
        all the neurons in that previous layer. Then applies leaky ReLU activation function. 
        """
        # Multiply the inputs by the weights, identical to: z = np.dot(sensory_inputs, weights) 
        z = self.W(sensory_inputs)
        # Store z for later use
        self.z = z  
        # Apply leaky relu function
        return F.leaky_relu(z, negative_slope= self.leaky_slope)

    def firing_rate(self, sensory_inputs, c_n, dynamic_step=False, beta=1.0):
        """
        Calculates the final firing rate of the neuron by combining the bottom-up 
        drive with the top-down dendritic modulation.

        - If dynamic_step=False: Returns the instantaneous target rate (for Backprop/Baseline).
        - If dynamic_step=True: Leaky-integrates the current state towards the target rate (for PID).

        """
        # 1. Get the bottom-up activation
        phi_z = self.bottom_up_proc(sensory_inputs)
        
        # 2. Get the top-down apical activation
        q_c = self.dendritic_proc(c_n)
        
        # 3. Combine them with multiplicative effect (element-wise multiplication)
        target_activation = (beta * q_c) * phi_z 
        
        # 4. Instantaneous vs. Dynamical Return
        if not dynamic_step:
            # Return the target activation directly, instantaneously
            return target_activation
 
        else:
            # The dynamics integrator continually grabs the current physical state (self.a_controlled) 
            # and nudges it step-by-step toward the target firing rate (target_r). 
            # If this is the first dynamic step, initialize a_controlled as baseline
            if self.a_controlled is None:
                    # Start physical settling from the baseline prediction
                    self.a_controlled = self.a_baseline.clone()
        
            # Step the physics forward 
            next_activation, _ = self.dynamics.step(self.a_controlled, target_activation)
            return next_activation
        
    def repolarize(self):
        """
        Clears the physical state memory between settling phases,
        returning the neuron to its baseline state for the next sensory input.
        """
        self.a_controlled = None  # type: ignore
    
    def get_bottom_up_activation_derivatives(self): 
        """
        This function calculates the derivative of the bottom-up (Leaky ReLU) activation function.
            If z>0, the output is z. (The slope/derivative is 1).
            If z≤0, the output is 0.01⋅z. (The slope/derivative is 0.01).

        When calculating the chain rule to pass your top-down control signal backward, 
        you must multiply the signal by this derivative.
        """
        if self.z is None:
            raise RuntimeError("Cannot compute derivative: forward pass hasn't occurred yet - z is still None.")
        derivative = torch.ones_like(self.z) # Start with a tensor of ones
        derivative[self.z <= 0] = self.leaky_slope # Set the slope to leaky_slope where z <= 0 using masking
        return derivative
    
    

    