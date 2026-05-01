import torch
import os 
import sys

# ==========================================
# TURN ON ANOMALY DETECTION FOR DEBUGGING
# It watches every single mathematical operation in real-time. 
# When a tensor becomes NaN or Inf, PyTorch instantly freezes the program and throws an error.
# torch.autograd.set_detect_anomaly(True)
# ==========================================

from models.network import Network
from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from xor.dataset import get_dataloader
from core.trainer import Trainer
from utils.utils import set_all_seeds, save_experiment, get_weight_metrics
from IPython.display import clear_output
from utils.utils import get_logger

logger = get_logger()

from tqdm import tqdm
import wandb

def main():
    set_all_seeds(7)
    task = "xor"  # Define the task (for documentation and saving purposes)
    epochs = 320

    wandb_on = True  # Set to True to enable W&B logging

    # 1. Initialize Anatomy (e.g., 2 inputs -> 4 hidden -> 1 output)
    network = Network(pop_sizes=[2, 4, 1])

    # 2. Initialize Mechanics
    controller = ControlMechanism(mode='pid', lr_c=0.1, max_steps=60)
    plasticity = Plasticity(lr_theta=0.2)

    # 3. Initialize variables, pbar and objects for training
    trainer = Trainer(network, controller, plasticity)
    dataloader = get_dataloader(batch_size=4, shuffle=True)

    best_loss = float('inf')
    current_avg_loss = float('inf')

    if wandb_on:
        wandb.init(project="target-learning", name=f"{task}_{controller.mode}_training_run")
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(range(epochs), desc="Learning")

    # 4. Train
    for epoch in progress_bar:
        current_avg_loss, avg_control_mag = trainer.train_one_epoch(dataloader)

        # Checkpoint: Save if this is the best model so far
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss
            save_experiment(
                network=network, 
                controller=controller, 
                plasticity=plasticity, 
                epoch=epoch, 
                loss=current_avg_loss, 
                task=task
            )

        progress_bar.set_postfix({
            "Loss": f"{current_avg_loss:.4f}", 
            "Best": f"{best_loss:.4f}"
        })

        if wandb.run is not None:
            wandb_metrics = {
            "Training Loss": current_avg_loss,
            "Control Magnitude": avg_control_mag
            }
            wandb_metrics.update(get_weight_metrics(network))
            wandb.log(wandb_metrics, step=epoch)
        elif wandb_on: 
            logger.warning("W&B logging is enabled but no active run found. Metrics will not be logged to W&B.")

    print(f"\n✅ Training Complete! Best model saved with loss {best_loss:.4f}")

    # Cleanly close the W&B run
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()