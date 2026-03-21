import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronEnsemble(nn.Module):
    def __init__(self, num_inputs, num_neurons):
        """
        Builds the anatomy of the neuron ensemble.
        """
        super().__init__()
        self.num_neurons = num_neurons
        
        # 1. HOLDING WEIGHTS AND BIASES
        self.W = nn.Linear(num_inputs, num_neurons)
        
        # 2. INITIALIZING STATE MEMORY 
        self.a_baseline = None # Activation for the first guess
        self.a_controlled = None # The dynamic physical state that changes over time given the control signal (top-down input)
    
    def dendritic_proc(self, c_n):
        # sigmoid function
        return torch.sigmoid(c_n)

    def sensory_proc(self, sensory_inputs):
        """
        Gets the sensory inputs and sums them up (z_n). The total bottom-up input for our current neuron (neuron n) is 
        the activation of a neuron from the previous layer (neuron m) times their connecting weight (wmn​), summed up across 
        all the neurons in that previous layer. Then applies leaky ReLU activation function. 
        """
        # Multiply the inputs by the weights, then add the biases.
        # Identical to: z = np.dot(sensory_inputs, weights) + biases
        z = self.W(sensory_inputs) 
        # Apply leaky relu function
        return F.leaky_relu(z, negative_slope=0.01)

    def firing_rate(self, sensory_inputs, c_n, beta=1.0):
        """
        Calculates the final firing rate of the neuron by combining the bottom-up 
        sensory drive with the top-down dendritic modulation.
        """
        # 1. Get the bottom-up sensory activation
        phi_z = self.sensory_proc(sensory_inputs)

        # 2. Get the top-down apical activation
        q_c = self.dendritic_proc(c_n)
        
        # 3. Combine them with multiplicative effect (element-wise multiplication)
        r = (beta * q_c + 1) * phi_z 
        
        return r

class ContinuousTimeStepper:
    def __init__(self, dt=0.1, tau=1.0, epsilon=1e-4):
        self.dt = dt
        self.tau = tau
        self.epsilon = epsilon # Convergence threshold

    def step(self, a_current, a_target):
        """
        Calculates the next state of activation.
        a_current: a_controlled at time t
        r_target: the modulated firing rate phi(z_n, c_n)
        """
        # 1. Activation change per time step based on the difference between current activation and target firing rate
        delta_a = (self.dt / self.tau) * (a_target - a_current)

        # 2. Update the activation for the next time step
        a_next = a_current + delta_a

        # 3. Check for convergence (if the change in activation is smaller than the threshold)
        is_converged = torch.norm(delta_a) < self.epsilon
        
        return a_next, is_converged
        
        
    
    

class Plasticity:
    """
    Updates the weights based on the learning rule
    """
    def learning_rule(a_pre, a_baseline, a_controlled):  
        errors = a_controlled - a_baseline # one for each neuron
        return np.dot(a_pre.T, errors) # a_pre.shape is a single row (1, k) and errors.shape is (1, u)
    
    def train_single_step(self): 
        # TODO
        return 
        