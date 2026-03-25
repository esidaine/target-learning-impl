import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath('..'))
from src.utils.utils import get_logger

logger = get_logger()


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

        # Turn on evaluation mode (important when we have dropout or batchnorm layers, 
        # which we don't in this simple model, but good practice for future extensions)
        network.eval() 

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
        
        # Turn on training mode (important for dropout or batchnorm layers, 
        # which we don't have, but good practice for future extensions)
        network.train()

        if self.mode == 'backprop':
            return self._optimize_via_backprop(control_signals, sensory_inputs, target_y, network)
        elif self.mode == 'pid':
            return self._optimize_via_pid(control_signals, sensory_inputs, target_y, network)
        else:
            raise ValueError("Mode must be 'backprop' or 'pid'")
        
    def _optimize_via_backprop(self, control_signals, sensory_inputs, target_y, network):
        
        # Use adam to optimize c_n with momentum and adaptive learning rates.
        c_optimizer = torch.optim.Adam(control_signals, lr=self.lr_c)

        # Loss function (Mean Squared Error denotes averaging over the errors f(x) - y squared (per data point) across the batch)
        criterion = nn.MSELoss()
        
        for step in range(self.max_steps):
            # 1. Zero the gradients for c_n
            c_optimizer.zero_grad()
            
            # 2. Forward pass with the current control signals (note that this is the second pass, since the baseline pass has c_n = 0)
            # This builds the computational graph connecting c_n to the output
            y_pred = network(sensory_inputs, control_signals=control_signals, save_baseline=False)

            # 3. Calculate how far off we are from the target
            loss = criterion(y_pred, target_y)

            # 4. Check for early convergence
            if loss.item() < self.tolerance:
                logger.info(f"Converged at step {step} with loss {loss.item()}")
                break

            # 5. Backpropagate the error to calculate gradients with respect to c_n
            loss.backward()
            
            # 6. Take a step to update c_n
            c_optimizer.step()

        # Clean up memory 
        for pop in network.populations:
            if pop.a_controlled is not None:
                pop.a_controlled = pop.a_controlled.detach()

        # Return the optimized control signals (c*)
        # We detach them because we are done optimizing them and don't want to carry the computational graph forward.
        return [c_n.detach() for c_n in control_signals]
       
    @torch.no_grad() # Turn off PyTorch autograd for PID
    def _optimize_via_pid(self, c_n, sensory_inputs, target_y, network):
        #TODO
        return c_n

        




