import torch
import torch.nn as nn
from utils.utils import get_logger
from core.euler_integrators import ControlErrorIntegrator
from visualizations.vis_helpers import make_manim_snapshot
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.functional as F
import numpy as np

logger = get_logger()

@dataclass
class OptimizationMetrics:
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    steps_taken: int = 0
    converged: bool = False
    loss_history: list[float] = field(default_factory=list)
    final_control: Optional[torch.Tensor] = None  # last global u, PID only

    # Stores a list of time steps. Each time step contains a list of layer activations.
    state_history: list[list[np.ndarray]] = field(default_factory=list)

    # ---- write API ---------------------------------------------------
    def record(self, loss: float) -> None:
        """Log a loss measurement; first call sets initial_loss."""
        if self.initial_loss is None:
            self.initial_loss = loss
        self.final_loss = loss
        self.loss_history.append(loss)
    
    def step(self) -> None:
        self.steps_taken += 1

    def mark_converged(self) -> None:
        self.converged = True

    # ---- read API ----------------------------------------------------
    @property
    def improvement(self) -> float:
        if self.initial_loss is None or self.final_loss is None:
            return 0.0
        return self.initial_loss - self.final_loss

    @property
    def improved(self) -> bool:
        return self.improvement > 0


class ControlMechanism: 
    """
    Generates the control signal (c_n) (per neuron) and finds the optimal control signal (c_n*) that allows the neuron to converge to the target firing rate (a_target).
    Has two implementations: with backpropagation and with PID control 
    """
    def __init__(self, mode='backprop', lr_c=0.1, momentum=0.5, max_steps=100, 
             dt=0.1, tau=1.0, alpha=0.01, k_p=0.8, use_derivative=True):
        self.mode = mode
        self.lr_c = lr_c
        self.momentum = momentum
        self.max_steps = max_steps
        self.tolerance = 0.001
        self.dt = dt
        self.tau = tau
        self.alpha = alpha
        self.k_p = k_p
        self.use_derivative = use_derivative

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
        metrics = OptimizationMetrics()

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

        metrics.record(F.mse_loss(baseline_pred, target_y).item())


        # ==========================================
        # STEP 2: INITIALIZE CONTROLS TO BE TUNED
        # ==========================================
        # Create the list of c_n tensors that will be tuned and where gradients will be tracked.
        # Each batch of sensory inputs will have its own control signal for each population (first hidden layer, second hidden layer, etc.)
        initial_controls = self.initialize_controls(batch_size, network.populations)
    
        # ==========================================
        # STEP 3: THE OPTIMIZATION PHASE
        # ==========================================
        
        # Turn on training mode (important for dropout or batchnorm layers, 
        # which we don't have, but good practice for future extensions)
        network.train()

        if self.mode == 'backprop':
            optimized_controls = self._optimize_via_backprop(
                initial_controls, sensory_inputs, target_y, network, metrics
                )
        elif self.mode == 'pid':
            optimized_controls = self._optimize_via_pid(
                initial_controls, sensory_inputs, target_y, network, baseline_pred, metrics, self.use_derivative
            )
        else:
            raise ValueError("Mode must be 'backprop' or 'pid'")
        

        if not metrics.improved:
            logger.warning(
                f"Control optimization did not improve loss: "
                f"{metrics.initial_loss:.6f} -> {metrics.final_loss:.6f} "
                f"in {metrics.steps_taken} steps"
            )

        
        return optimized_controls, metrics
        
    def _optimize_via_backprop(self, control_signals, sensory_inputs, target_y, network, metrics):
        
        c_optimizer = torch.optim.SGD(control_signals, lr=self.lr_c, momentum=self.momentum)

        # Loss function (Mean Squared Error denotes averaging over the errors f(x) - y squared (per data point) across the batch)
        criterion = nn.MSELoss()
        
        for _ in range(self.max_steps):

            # 1. Zero the gradients for c_n
            c_optimizer.zero_grad()
            
            # 2. Forward pass with the current control signals (note that this is the second pass, since the baseline pass has c_n = 0)
            # This builds the computational graph connecting c_n to the output
            y_pred = network(sensory_inputs, control_signals=control_signals, save_baseline=False)

            # 3. Calculate how far off we are from the target
            loss = criterion(y_pred, target_y)

            if loss.item() < self.tolerance:
                metrics.mark_converged()
                break

            # 4. Backpropagate the error to calculate gradients with respect to c_n
            # Note that we multiply the output error by the transpose of the weights (W.T) - weight symmetry!
            loss.backward()
            
            # 5. Take a step to update c_n
            c_optimizer.step()
            metrics.step()
        
            # 6. Record the new loss after this step
            with torch.no_grad():
                post_step_pred = network(sensory_inputs, control_signals=control_signals, save_baseline=False)
                metrics.record(criterion(post_step_pred, target_y).item())

        # Clean up memory 
        for pop in network.populations:
            if pop.a_controlled is not None:
                pop.a_controlled = pop.a_controlled.detach()

        # Return the optimized control signals (c*)
        # We detach them because we are done optimizing them and don't want to carry the computational graph forward.
        return [c_n.detach() for c_n in control_signals]
       
    @torch.no_grad() # Turn off PyTorch autograd for PID. That means we won't use W.T for the feedback
    def _optimize_via_pid(self, control_signals, sensory_inputs, target_y, network, baseline_pred, metrics, use_derivative):
        """
        The controller pushes, the neurons move, the controller checks the new output, 
        and pushes again. This happens continuously over your max_steps loop. 
        It is a dynamical system settling into an equilibrium.
        The controller knows the global error. This is multiplied by the feedback weights,
        shattering the global error into thousands of specific, localized control signals (cn​),
        to every single hidden neuron simultaneously.
        The hidden neurons have no idea what the global error is, but they do know their own control signal/ apical signal. 
    
        """
        manim_snapshot = False # Set to True to enable Manim snapshots during PID optimization 

        control_stepper = ControlErrorIntegrator(dt=self.dt, tau=self.tau, alpha=self.alpha, k_p=self.k_p)

        # Track the global error of the output layer 
        batch_size = sensory_inputs.size(0)
        output_size = target_y.size(1)

        global_control_integral = torch.zeros(batch_size, output_size, device=sensory_inputs.device)

        # Initialize y_pred with the control-free baseline measurement we already took
        # shape [batch_size, num_output_neurons]
        y_pred = baseline_pred

        # control signals were created with required_grad=True, so detach for safety
        local_controls = [c.detach() for c in control_signals]
        
        # Initialize the global control signal to zeros (shape [batch_size, num_output_neurons])
        global_control = torch.zeros(batch_size, output_size, device=sensory_inputs.device)

        # Dynamic inversion: 
        # Finding the ideal activation state for each neuron to match the final output target
        # It finds this state incrementally by nudging the control signal over time until the physical simulation settles
        for _ in range(self.max_steps):
            # ==========================================
            # EARLY EXIT
            # ==========================================
            if metrics.final_loss is not None and metrics.final_loss < self.tolerance:
                metrics.mark_converged()
                if metrics.steps_taken == 0:
                    # Ensure a_controlled is initialized for downstream consumers.
                    network(sensory_inputs, control_signals=local_controls,
                            save_baseline=False, dynamic_step=True)
                break
            
            # ==========================================
            # UPDATE GLOBAL CONTROL SIGNAL, STEP IN TIME
            # ==========================================
            output_error = target_y - y_pred
            global_control_integral, global_control = control_stepper.step(global_control_integral, output_error) 

            # ==========================================
            # CREDIT ASSIGNMENT, PASS THE GLOBAL CONTROL BACKWARD 
            # ==========================================
            local_controls = network.DFC_project_feedback(global_control, use_derivative=use_derivative)
            # local_controls = network.chain_rule_project_feedback(global_control) 

            # ==========================================
            # SIMULATE A FORWARD PASS 
            # ==========================================
            y_pred = network(sensory_inputs, control_signals=local_controls, save_baseline=False, dynamic_step=True)
            metrics.step()

            # ==========================================
            # 6. RECORD AGAIN
            # ==========================================
            metrics.record(F.mse_loss(y_pred, target_y).item())

            # ==========================================
            #  SNAPSHOTTING HOOK FOR VISUALIZATION
            # ==========================================
     
            if manim_snapshot:
                make_manim_snapshot(network, local_controls, metrics)

        metrics.final_control = global_control  # save last u for diagnostics

        return local_controls
   

        




