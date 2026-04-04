import os
import sys

#sys.path.append(os.path.abspath('..'))
from src.models.network import Network
from src.core.controllers import ControlMechanism
from src.core.plasticity import Plasticity
from src.xor.dataset import get_dataloader
from src.core.trainer import Trainer
from src.utils.utils import set_all_seeds, save_experiment
from IPython.display import clear_output

from tqdm import tqdm
import wandb

def main():
    set_all_seeds(7)
    task = "xor"  # Define the task (for documentation and saving purposes)
    epochs = 550

    wandb_on = True  # Set to True to enable W&B logging

    # 1. Initialize Anatomy (e.g., 2 inputs -> 4 hidden -> 1 output)
    network = Network(pop_sizes=[2, 4, 1])

    # 2. Initialize Mechanics
    controller = ControlMechanism(mode='backprop', lr_c=0.1, max_steps=60)
    plasticity = Plasticity(lr_theta=0.2)

    # 3. Initialize variables, pbar and objects for training
    trainer = Trainer(network, controller, plasticity)
    dataloader = get_dataloader(batch_size=4, shuffle=True)

    best_loss = float('inf')
    current_avg_loss = float('inf')

    if wandb_on:
        wandb.init(project="target-learning", name=f"{task}_{controller.mode}_training_run")
        print("W&B Initialized! Starting training loop...")
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(range(epochs), desc="Learning")

    # 4. Train
    for epoch in progress_bar:
        current_avg_loss, avg_control_mag = trainer.train_one_epoch(dataloader, epoch)

        # Checkpoint: Save if this is the best model so far
        best_loss = current_avg_loss
        save_experiment(
            network=network, 
            controller=controller, 
            plasticity=plasticity, 
            epoch=epoch, 
            loss=current_avg_loss, 
            task=task, 
            is_best=True    # This overwrites the "best_model" file
        )

        progress_bar.set_postfix({
            "Loss": f"{current_avg_loss:.4f}", 
            "Best": f"{best_loss:.4f}"
        })

        if wandb_on:
            wandb.log({"Training Loss": current_avg_loss, "Control Magnitude": avg_control_mag}, step=epoch)

    save_experiment(network, controller, plasticity, epochs, current_avg_loss, task, is_best=False)
    print(f"\n✅ Training Complete! Best model saved with loss {best_loss:.4f}")

    # Cleanly close the W&B run
    wandb.finish()

if __name__ == "__main__":
    main()