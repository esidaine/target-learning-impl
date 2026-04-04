# Weights & Biases (wandb.ai).
# Instead of manually trying to draw graphs of your network's error rate, wandb is a 
# free library you import that silently watches your network train and builds a live, beautiful, 
# interactive dashboard in your web browser. You can literally watch your Target Learning PID 
# controller drop the error rate in real-time. It is the absolute gold standard for ML research papers today.

import torch
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

    def train_one_epoch(self, dataloader, epoch_idx)-> tuple[float, float]:
        self.network.train() # mark training mode 
        epoch_loss = 0.0

        # Track the magnitude of the control signal across epochs to monitor how much the controller is having to intervene.
        # If this goes down, it means the network is learning to produce the correct output on its own
        # without needing as much top-down control.
        epoch_control_magnitude = 0.0 
        
        for batch_idx, (sensory_inputs, target_y) in enumerate(dataloader):

            # ==========================================
            # 1. THE BASELINE & CONTROL PHASE
            # ==========================================
            # This single call does the free pass, saves 'a_baseline', 
            # tunes 'c_n', and saves 'a_controlled' in all populations.
            optimal_controls = self.controller.optimize_control_signal(
                sensory_inputs=sensory_inputs, 
                target_y=target_y, 
                network=self.network
            )


            # Calculate the mean absolute magnitude of the control signals across all layers
            with torch.no_grad():
                batch_c_mag = sum(torch.abs(c).mean().item() for c in optimal_controls) / len(optimal_controls)
                epoch_control_magnitude += batch_c_mag

            # ==========================================
            # 2. THE PLASTICITY PHASE
            # ==========================================
            # The weights are updated based on the difference between a_controlled and a_baseline.
            self.plasticity.update_weights(
                network=self.network, 
                sensory_inputs=sensory_inputs
            )

            # ==========================================
            # 3. MONITORING 
            # ==========================================
            # How wrong the network's initial guess was.
            # We grab the baseline activation from the very last layer (spacially).
            baseline_predictions = self.network.populations[-1].a_baseline
            
            # Calculate the loss 
            # Is the network actually getting the predictions right?
            # When the baseline activation matches a^c*, then we have 
            # successfully forced the network to produce the correct prediction y
            loss = self.criterion(baseline_predictions, target_y)
            epoch_loss += loss.item()

        # Calculate the average loss and control magnitude for this epoch
        # len(dataloader) is the number of batches with Total Batches = Total Samples / Batch Size
        average_loss = epoch_loss / len(dataloader)
        average_control_mag = epoch_control_magnitude / len(dataloader)

        return average_loss, average_control_mag