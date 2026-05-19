import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_logger
from typing import Optional
from core.euler_integrators import FiringRateDynamics

logger = get_logger()


class Network(nn.Module):
    def __init__(self, pop_sizes, dendritic_effect="additive"):
        super().__init__()
        
        # Note that pop[0] refers to the first hidden layer, not the input layer
        self.populations: nn.ModuleList = nn.ModuleList([
            NeuralPopulation(
                num_inputs = pop_sizes[i], 
                num_neurons = pop_sizes[i+1], 
                output_dim = pop_sizes[-1],
                dendritic_effect=dendritic_effect
            )
            for i in range(len(pop_sizes) - 1)
        ])

        #### 
        # DEFINE Q WEIGTHT MATRIX AS TRANSPOSE OF FORWARD WEIGHTS
        ####

    def project_feedback(self, global_control: torch.Tensor) -> list[torch.Tensor]:
            """
            DFC credit-assignement
            DFC uses parallel broadcasting, where the global controller signal $u$ is sent directly to all hidden
            layers simultaneously via a feedback matrix
            -------
            list[Tensor]
                `[c_1, c_2, ..., c_L]`, where `c_i = Q_i @ u`, one per layer/population.
                Where u is the global control signal produced by the PID controller. It has one number per neuron

            u                 has shape (batch, output_dim)
            Q_i               has shape (num_neurons_in_layer_i, output_dim)
            c_i = Q_i @ u     has shape (batch, num_neurons_in_layer_i)

            """
            L = len(self.populations) 
        
            # Pre-allocate a list of size L to avoid the IndexError
            local_controls = [torch.empty(0)] * L 
            
            for i, pop in enumerate(self.populations):
                # Pylance checks
                assert isinstance(pop, NeuralPopulation)
                assert isinstance(pop.Q, nn.Linear)
                
                local_controls[i] = F.linear(global_control, pop.Q.weight)
    
            return local_controls


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
    def __init__(self, num_inputs: int, num_neurons: int, output_dim: int, dendritic_effect: str, leaky_slope: float = 0.01):
        """
        Represents a group of Multi-Compartment neurons with specific anatomy and processing rules of top-down and bottom-up input.

        - W : forward weights, shape (num_neurons, num_inputs).
          Used in the forward pass.  Frozen during control optimisation;
          updated separately by the plasticity rule.

        - Q : DFC feedback weights, shape (num_neurons, output_dim).
          Maps the *global* controller signal u into a *local* control
          signal c_i for this layer.  Independent of W. Fixed throughout
          training (DFC-fixed).  Flip `requires_grad` to learn it (DFC-SS).

        """
        super().__init__()
        self.num_neurons = num_neurons
        self.output_dim = output_dim
        self.leaky_slope = leaky_slope
        self.dendritic_effect = dendritic_effect
        
        # Forward weights, disable biases and gradient tracking
        self.W = nn.Linear(num_inputs, num_neurons, bias=True)
        self.W.weight.requires_grad = False
        self.W.bias.requires_grad = False
        
        # State memory
        self.a_baseline: Optional[torch.Tensor] = None # Activation for the first guess
        self.z: Optional[torch.Tensor] = None # Bottom Up 
        self.a_controlled: Optional[torch.Tensor] = None # The dynamic physical state that changes over time given the control signal (top-down input)
        self.firing_settled: Optional[bool] = None # Diagnostic variable to track whether the dynamics have settled at the target yet. 
        self.target_activation: Optional[torch.Tensor] = None 

        # Internal dynamics for the firing rate to evolve over time
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
        # It is the weighted sum of presynaptic activities, before the nonlinearity
        z = self.W(sensory_inputs)
        # Store z for later use
        self.z = z  
        # Apply leaky relu function
        return F.leaky_relu(z, negative_slope=self.leaky_slope)

    def firing_rate(self, sensory_inputs, c_n, dynamic_step=False, beta=1.0):
        """
        Calculates the final firing rate of the neuron by combining the bottom-up 
        drive with the top-down dendritic modulation.

        - If dynamic_step=False: Returns the instantaneous target rate (for Backprop/Baseline).
        - If dynamic_step=True: Leaky-integrates the current state towards the target rate (for PID).

        """
        
        # 1. Get the bottom-up activation
        phi_z = self.bottom_up_proc(sensory_inputs)
    
        
        # 3. Combine them with multiplicative or additive effect (element-wise multiplication)
        if self.dendritic_effect == "multiplicative":
            # 2. Get the top-down apical activation
            # When c_n is very negative, q_c approaches 0 (silences the neuron).
            # When c_n is very positive, q_c approaches 2 (doubles the bottom-up rate).
            q_c = self.dendritic_proc(c_n)
            target_activation = (beta * q_c) * phi_z 
        elif self.dendritic_effect == "additive":
            # Combine additively: adds or subtracts at most 1.0 from the firing rate (capped by the tanh nonlinearity)
            target_activation = phi_z + torch.tanh(c_n)
        else:
            raise ValueError(f"Invalid dendritic_effect: {self.dendritic_effect}. Must be 'multiplicative' or 'additive'.")
        
        self.target_activation = target_activation
        
        # 4. Instantaneous vs. Dynamical Return
        if not dynamic_step:
            # Return the target activation directly
            return target_activation # Model A: instantaneous
 
        else:
            # The dynamics integrator continually grabs the current physical state (self.a_controlled) 
            # and nudges it step-by-step toward the target firing rate (target_r). 
            # If this is the first dynamic step, initialize a_controlled as baseline
            if self.a_controlled is None:
                    if self.a_baseline is None:
                        raise RuntimeError(
                            "Dynamic step requested before a baseline pass. "
                            "Call network.forward(..., save_baseline=True) first."
                        )
                    # Start physical settling from the baseline prediction
                    self.a_controlled = self.a_baseline.clone()
        
            # Step the physics forward 
            next_activation, settled = self.dynamics.step(self.a_controlled, target_activation)
            self.firing_settled = bool(settled.item())  # diagnostic only

            return next_activation # Model B: leaky-integrated
        
    def repolarize(self):
        """
        Clears the physical state memory between settling phases,
        returning the neuron to its baseline state for the next sensory input.
        """
        self.a_controlled = None  # type: ignore
    
    # THIS IS TO BE DELETED, MAY BE USED FOR TESTS
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
    
    

    