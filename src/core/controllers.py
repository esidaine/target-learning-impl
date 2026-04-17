import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath('..'))
from src.utils.utils import get_logger
from src.core.euler_integrators import ControlErrorIntegrator
from dataclasses import dataclass

logger = get_logger()

@dataclass
class OptimizationMetrics:
    initial_loss: float = float('inf')
    final_loss: float = float('inf')
    steps_taken: int = 0
    converged: bool = False


class ControlMechanism: 
    """
    Generates the control signal (c_n) (per neuron) and finds the optimal control signal (c_n*) that allows the neuron to converge to the target firing rate (a_target).
    Has two implementations: with backpropagation and with PID control 
    """
    def __init__(self, mode='backprop', lr_c=0.1, max_steps=50, 
                 dt=0.1, tau=1.0, alpha=0.1, k_p=1.0):
        
        # Determine which control algorithm to use
        self.mode = mode 
        
        # --- Gradient-Based Optimization Parameters ---
        self.lr_c = lr_c             # Learning rate for updating c_n
        self.max_steps = max_steps   # Max iterations to find c_n*
        self.tolerance = 0.001        # Convergence threshold [!] Should be reconsidered or tuned
        
        # --- PID Controller Parameters ---
        self.dt = 0.1
        self.tau = 1.0
        self.alpha = 0.1 # The leak (prevents integral from exploding to infinity)
        self.k_p = 1.0   # Proportional gain

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

        # Data goes in, flows through the frozen W, and makes a prediction.
        # We use torch.no_grad() because the baseline guess requires no optimization.
        # This prevents PyTorch from building a useless computational graph.
        with torch.no_grad():
            baseline_pred = network(sensory_inputs, control_signals=None, save_baseline=True)

        # ==========================================
        # STEP 2: INITIALIZE CONTROLS TO BE TUNED
        # ==========================================
        # Create the list of c_n tensors that will be tuned and where gradients will be tracked.
        # Each batch of sensory inputs will have its own control signal for each population (first hidden layer, second hidden layer, etc.)
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
            return self._optimize_via_pid(control_signals, sensory_inputs, target_y, network, baseline_pred)
        else:
            raise ValueError("Mode must be 'backprop' or 'pid'")
        
    def _optimize_via_backprop(self, control_signals, sensory_inputs, target_y, network):
        
        # Use adam to optimize c_n with momentum and adaptive learning rates.
        c_optimizer = torch.optim.Adam(control_signals, lr=self.lr_c)

        # Loss function (Mean Squared Error denotes averaging over the errors f(x) - y squared (per data point) across the batch)
        criterion = nn.MSELoss()
        metrics = OptimizationMetrics()
        
        for step in range(self.max_steps):
            metrics.steps_taken = step + 1

            # 1. Zero the gradients for c_n
            c_optimizer.zero_grad()
            
            # 2. Forward pass with the current control signals (note that this is the second pass, since the baseline pass has c_n = 0)
            # This builds the computational graph connecting c_n to the output
            y_pred = network(sensory_inputs, control_signals=control_signals, save_baseline=False)

            # 3. Calculate how far off we are from the target
            loss = criterion(y_pred, target_y)
            if step == 0:
                metrics.initial_loss = loss.item() # Store the initial loss for monitoring improvement
            metrics.final_loss = loss.item()

            # 4. Check for early convergence
            if metrics.final_loss < self.tolerance:
                logger.debug(f"Converged at step {step} with loss {loss.item()}")
                break

            # 5. Backpropagate the error to calculate gradients with respect to c_n
            # Note that we multiply the output error by the transpose of the weights (W.T) - weight symmetry!
            loss.backward()
            
            # 6. Take a step to update c_n
            c_optimizer.step()

        improvement = metrics.initial_loss - metrics.final_loss
        if metrics.steps_taken == self.max_steps and improvement <= 0.001: 
            logger.warning(f" [!] Control Optimization struggled!  {metrics.initial_loss:.4f} -> {metrics.final_loss:.4f} in {metrics.steps_taken} steps.")

        # Clean up memory 
        for pop in network.populations:
            if pop.a_controlled is not None:
                pop.a_controlled = pop.a_controlled.detach()

        # Return the optimized control signals (c*)
        # We detach them because we are done optimizing them and don't want to carry the computational graph forward.
        return [c_n.detach() for c_n in control_signals]
       
    @torch.no_grad() # Turn off PyTorch autograd for PID. That means we won't use W.T for the feedback
    def _optimize_via_pid(self, control_signals, sensory_inputs, target_y, network, baseline_pred):
        """
        The controller pushes (u), the neurons move (v or r), the controller checks the new output, 
        and pushes again. This happens continuously over your max_steps loop. 
        It is a dynamical system settling into an equilibrium.
        The controller knows the global error. This is multiplied by the feedback weights,
        shattering the global error into thousands of specific, localized control signals (cn​),
        to every single hidden neuron simultaneously.
        The hidden neurons have no idea what the global error is, but they do know their own control signal. 
    
        """
        metrics = OptimizationMetrics()

        control_stepper =  ControlErrorIntegrator(dt=0.1, tau=1.0, alpha=0.1, k_p=0.05) # Parameters need to be changed or reconsidered

        # Track the the global error of the output layer 
        batch_size = sensory_inputs.size(0)
        output_size = target_y.size(1)

        global_control = torch.zeros(batch_size, output_size) # This is the global control signal that we will update iteratively
        global_control_integral = torch.zeros(batch_size, output_size)

        # Initialize y_pred with the control-free baseline measurement we already took
        # shape [batch_size, num_output_neurons]
        y_pred = baseline_pred
        
        # Dynamic inversion: 
        # Finding the ideal activation state for each neuron to match the final output target
        # It finds this state incrementally by nudging the control signal over time until the physical simulation settles
        for step in range(self.max_steps):
            metrics.steps_taken = step + 1

            # ==========================================
            # 1. CALCULATE GLOBAL ERROR 
            # ==========================================
            # Both y have the same shape, so subtracting them gives us a vector of raw errors
            # for each output neuron, for each data point in the batch.
            output_error = target_y - y_pred 

            # ==========================================
            # 2. EARLY EXIT
            # ==========================================
            mse_loss = torch.mean(output_error ** 2).item()
            if step == 0:
                metrics.initial_loss = mse_loss
            metrics.final_loss = mse_loss
            
            if mse_loss < self.tolerance:
                logger.debug(f"PID Converged at step {step} with loss {mse_loss}")
                break
            
            # ==========================================
            # 3. UPDATE GLOBAL CONTROL SIGNAL, STEP IN TIME
            # ==========================================
            global_control_integral, global_control = control_stepper.step(global_control, global_control_integral, output_error) 

            # ==========================================
            # 4. CREDIT ASSIGNMENT, PASS THE GLOBAL CONTROL BACKWARD 
            # ==========================================
            local_controls = network.get_local_controls(global_control)

            # ==========================================
            # 5. SIMULATE A FORWARD PASS 
            # ==========================================
            y_pred = network(sensory_inputs, control_signals=local_controls, save_baseline=False)

        return global_control
   

        




