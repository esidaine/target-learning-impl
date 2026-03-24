import torch

class ContinuousTimeStepper:
    def __init__(self, dt=0.1, tau=1.0, epsilon=1e-4):
        self.dt = dt
        self.tau = tau
        self.epsilon = epsilon # Convergence threshold

    def step(self, a_current, a_target):
        """
        Calculates the next state of activation.
        a_current: a_controlled at time t
        a_target: the modulated firing rate phi(z_n, c_n) to optimality
        """
        # 1. Change of Neural Activation per time step based on the difference between current activation and target firing rate
        delta_a = (self.dt / self.tau) * (a_target - a_current)

        # 2. Update the activation for the next time step
        a_next = a_current + delta_a

        # 3. Check for convergence (if the change in activation is smaller than the threshold)
        is_converged = torch.norm(delta_a) < self.epsilon
        
        return a_next, is_converged