import torch

class FiringRateDynamics:
    """
    Leaky integrator that updates the neural activation 'a_n' at each time step based on the difference between the current activation and the target firing rate 'phi(z_n, c_n)'.
    Continuous differential equation for the neurons' membrane voltage
    """
    def __init__(self, dt=0.1, tau=1.0, epsilon=1e-4):
        self.dt = dt
        self.tau = tau # A high tau makes the system more slow in settling 
        self.epsilon = epsilon # Convergence threshold

    def step(self, current_a, target_a):
        """
        Calculates the next state of activation.
        v_current: v_controlled at time t
        v_target: the modulated firing rate phi(z_n, c_n) to optimality
        """
        # 1. Change of Neural Activation per time step based on the difference between current activation and target firing rate
        delta_a = (self.dt / self.tau) * (target_a - current_a)

        # 2. Update the activation for the next time step
        a_next = current_a + delta_a

        # 3. Check for convergence (if the change in activation is smaller than the threshold)
        is_converged = torch.norm(delta_a) < self.epsilon
        
        return a_next, is_converged
    

class ControlErrorIntegrator:
    """
    Integrates the control error over time, with a leakage term (alpha) 
    to prevent the integral from exploding to infinity.
    """
    def __init__(self, dt=0.1, tau=1.0, alpha=0.1, k_p=0.05):
        self.dt = dt
        self.tau = tau # ! Neuron tau and controller tau need to be aligned
        self.alpha = alpha
        self.k_p = k_p # Proportional gain 

    def step(self, current_controls, current_controls_integral, error):
        # The leak term alpha * current_integral causes the control signal to drop, 
        # which causes the error to increase. The leak is the teacher actively trying to walk away. 
        # It penalizes the control signal, forcing it to be as close to zero as possible. 

        # 1. Look at the current error and add it to your memory
        # The leak term keeps the integral from growing indefinitely, which is important for stability.
        delta_integral = (self.dt / self.tau) * (error - self.alpha * current_controls)

        # 2. Update the integral memory with the new error contribution
        next_controls_integral = current_controls_integral + delta_integral

        # 3. Calculate the new control signal 
        # We add (k_p * error) to our updated memory.
        next_controls = next_controls_integral + self.k_p * error
        return next_controls_integral, next_controls