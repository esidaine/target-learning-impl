import numpy as np 

class NeuronEnsemble:
    def __init__(self, num_inputs, num_neurons):
        """
        Builds the anatomy of the neuron ensemble.
        """
        self.num_neurons = num_neurons
        
        # 1. HOLDING WEIGHTS AND BIASES

        # Initializing the weights
        std_dev = np.sqrt(2.0 / num_inputs) # std_dev is the square root of the variance, 2.0 stops the signal from dying when ReLU chops it in half. 
        self.weights = np.random.randn(num_inputs, num_neurons) * std_dev  

        # Biases are initialized to zero. We need one for each neuron.
        self.biases = np.zeros(num_neurons)
        
        # We will also create an empty array to store our sensory input (z_n) later
        self.z = np.zeros(num_neurons)

    def compute_bottom_up_input(self, sensory_inputs):
        """
        Gets the sensory inputs and sums them up (z_n). The total bottom-up input for our current neuron (neuron n) is 
        the activation of a neuron from the previous layer (neuron m) times their connecting weight (wmn​), summed up across 
        all the neurons in that previous layer. 
        """
        # 2. CALCULATE z_n
        # Multiply the inputs by the weights, then add the biases.
        self.z = np.dot(sensory_inputs, self.weights) + self.biases
    

class Plasticity:
    """
    Updates the weights based on the learning rule
    """
    def learning_rule(a_pre, a_baseline, a_controlled):  
        errors = a_controlled - a_baseline # one for each neuron
        return np.dot(a_pre.T, errors) # a_pre.shape is a single row (1, k) and errors.shape is (1, u)
    
    def train_single_step(): 
        # TODO
        return 
        