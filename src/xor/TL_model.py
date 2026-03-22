import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, pop_sizes):
        super().__init__()
        
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
        activation = sensory_inputs # make sure activation is not unbound 

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

    def sensory_proc(self, sensory_inputs):
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
        sensory drive with the top-down dendritic modulation.
        """
        # 1. Get the bottom-up sensory activation
        phi_z = self.sensory_proc(sensory_inputs)

        # 2. Get the top-down apical activation
        q_c = self.dendritic_proc(c_n)
        
        # 3. Combine them with multiplicative effect (element-wise multiplication)
        r = (beta * q_c + 1) * phi_z 
        
        return r
    
class ControlMechanism: 
    """
    Generates the control signal (c_n) (per neuron) and finds the optimal control signal (c_n*) that allows the neuron to converge to the target firing rate (a_target).
    Has two implementations: with backpropagation and with PID control 
    """
    def __init__(self, mode='backprop', lr_c=0.1, max_steps=50):
        
        # Determine which control algorithm to use
        self.mode = mode 
        
        # --- Gradient-Based Optimization Parameters ---
        self.lr_c = lr_c             # Learning rate for updating c_n
        self.max_steps = max_steps   # Max iterations to find c_n*
        self.tolerance = 1e-4        # Convergence threshold
        
        # --- PID Controller Parameters ---
        #TODO

    def initialize_controls(self, batch_size, neuron_populations):
        """
        Creates the tunable 'c_n' tensors, one for each hidden layer/ group of neurons, stored in a list.
        """
        control_signals = [] 
        
        for pop in neuron_populations:
            c_n = torch.zeros(
                (batch_size, pop.num_neurons), 
                device=pop.W.weight.device, 
                requires_grad=True
            )
            
            control_signals.append(c_n) 
            
        return control_signals

    def optimize_control_signal(self, sensory_inputs, target_y, network):
        """
        Finds the optimal c_n* using the selected mode.
        """
        # Initialize c_n to zeros (neutral dendritic input)
        batch_size = sensory_inputs.size(0) # sensory_inputs is a tensor of shape [batch_size, number of features per data point]
        
         # ==========================================
        # STEP 1: THE BASELINE PASS
        # ==========================================
        # We use torch.no_grad() because the baseline guess requires no optimization.
        # This prevents PyTorch from building a useless computational graph.
        with torch.no_grad():
            network(sensory_inputs, control_signals=None, save_baseline=True)

        # ==========================================
        # STEP 2: INITIALIZE CONTROLS TO BE TUNED
        # ==========================================
        # Create the list of c_n tensors that will be tuned and where gradients will be tracked.
        control_signals = self.initialize_controls(batch_size, network.populations)

        # ==========================================
        # STEP 3: THE OPTIMIZATION PHASE
        # ==========================================
        if self.mode == 'backprop':
            return self._optimize_via_backprop(control_signals, sensory_inputs, target_y, network)
        elif self.mode == 'pid':
            return self._optimize_via_pid(control_signals, sensory_inputs, target_y, network)
        else:
            raise ValueError("Mode must be 'backprop' or 'pid'")
        
    def _optimize_via_backprop(self, c_n, sensory_inputs, target_y, network):
        
        # Use adam to optimize with momentum and adaptive learning rates.
        # Note: Cast to a list() for PyTorch stability. 
        c_optimizer = torch.optim.Adam(list(control_signals.values()), lr=self.lr_c)
       
        
    def _optimize_via_pid(self, c_n, sensory_inputs, target_y, network):
        #TODO
        return c_n
    

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
        