import torch

class ContinuousTimeStepper:
    """
    General leaky integrator that updates the neural activation 'a_n' at each time step based on the difference between the current activation and the target firing rate 'phi(z_n, c_n)'.
    """
    def __init__(self, dt=0.1, tau=1.0, epsilon=1e-4):
        self.dt = dt
        self.tau = tau
        self.epsilon = epsilon # Convergence threshold

    def step(self, voltage, target_voltage):
        """
        Calculates the next state of activation.
        a_current: a_controlled at time t
        a_target: the modulated firing rate phi(z_n, c_n) to optimality
        """
        # 1. Change of Neural Activation per time step based on the difference between current activation and target firing rate
        delta_a = (self.dt / self.tau) * (target_voltage - voltage)

        # 2. Update the activation for the next time step
        a_next = voltage + delta_a

        # 3. Check for convergence (if the change in activation is smaller than the threshold)
        is_converged = torch.norm(delta_a) < self.epsilon
        
        return a_next, is_converged
    

class ControlErrorIntegrator:
    """
    Integrates the control error over time, with a leakage term (alpha) 
    to prevent the integral from exploding to infinity.
    """
    def __init__(self, dt=0.1, tau=1.0, alpha=0.1):
        self.dt = dt
        self.tau = tau
        self.alpha = alpha

    def step(self, current_integral, error):
        delta_integral = (self.dt / self.tau) * (error - self.alpha * current_integral)
        return current_integral + delta_integral