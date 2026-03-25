# Weights & Biases (wandb.ai).
# Instead of manually trying to draw graphs of your network's error rate, wandb is a 
# free library you import that silently watches your network train and builds a live, beautiful, 
# interactive dashboard in your web browser. You can literally watch your Target Learning PID 
# controller drop the error rate in real-time. It is the absolute gold standard for ML research papers today.


import torch.nn as nn

class Trainer:
    def __init__(self, network, controller, plasticity):
        """
        The manager class that binds the Network, ControlMechanism, and Plasticity classes.
        We go through several loops of free phase, settling phase and weight-change phase. 
        Before each round we measure the baseline activation's error. If the network is learning, it decreases. 
        """
        self.network = network
        self.controller = controller
        self.plasticity = plasticity
        
        # Measures the baseline prediction where c_n​ = 0
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, dataloader, epoch_idx):
        self.network.train() # mark training mode 
        epoch_loss = 0.0
        
        for batch_idx, (sensory_inputs, target_y) in enumerate(dataloader):
            # ==========================================
            # 1. THE BASELINE & CONTROL PHASE
            # ==========================================
            # This single call does the free pass, saves 'a_baseline', 
            # tunes 'c_n', and saves 'a_controlled' in all populations.
            self.controller.optimize_control_signal(
                sensory_inputs=sensory_inputs, 
                target_y=target_y, 
                network=self.network
            )

            # ==========================================
            # 2. THE PLASTICITY PHASE
            # ==========================================
            # The weights are updated based on the difference between a_controlled and a_baseline.
            self.plasticity.train_single_step(
                network=self.network, 
                sensory_inputs=sensory_inputs
            )

            # ==========================================
            # 3. MONITORING 
            # ==========================================
            # We want to see how wrong the network's initial guess was.
            # We grab the baseline activation from the very last layer (spacially).
            baseline_predictions = self.network.populations[-1].a_baseline
            
            # Calculate the loss 
            # Is the network actually getting the predictions right?
            # When the baseline activation matches a^c*, then we have 
            # successfully forced the network to produce the correct prediction y
            loss = self.criterion(baseline_predictions, target_y)
            epoch_loss += loss.item()

            # len(dataloader) is the number of batches with Total Batches = Total Samples / Batch Size
            return epoch_loss / len(dataloader) # Average loss per batch for this epoch