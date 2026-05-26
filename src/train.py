import torch
import os 
import sys
import pickle
from pathlib import Path
from dataclasses import asdict

root = Path.cwd().parent           # target-learning-impl folder
for p in (root, root / 'src'):
    p = str(p)
    if p not in sys.path:
        sys.path.append(p)

# ==========================================
# TURN ON ANOMALY DETECTION FOR DEBUGGING
# It watches every single mathematical operation in real-time. 
# When a tensor becomes NaN or Inf, PyTorch instantly freezes the program and throws an error.
# torch.autograd.set_detect_anomaly(True)
# ==========================================

from models.network import Network
from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from data.xor.dataset import get_dataloader
from core.trainer import Trainer
from utils.utils import set_all_seeds, save_experiment, get_weight_metrics, get_logger
from utils.config import ExperimentConfig, BackpropControlParams, PIDControlParams, BackpropPlasticityParams, PIDPlasticityParams
from IPython.display import clear_output
from tqdm import tqdm
import wandb


logger = get_logger()


def main():
    config = ExperimentConfig(
        task="xor",
        mode="pid",             # Choose 'backprop' or 'pid'
        dendritic_effect="multiplicative", # Choose 'additive' or 'multiplicative'
        epochs=800,
        seed=7,
        pop_sizes=[2, 8, 1], 
        controller=PIDControlParams(),
        plasticity=PIDPlasticityParams()
    )


    set_all_seeds(config.seed)
    manim = False
    wandb_on = False  
    should_save = False

    print(f"🚀 {config.task.upper()} with {config.mode.upper()} ({config.dendritic_effect})")

    # Initialize Anatomy using config values
    network = Network(
        pop_sizes=config.pop_sizes,
        dendritic_effect=config.dendritic_effect
    )

    # Initialize Mechanics
    controller_kwargs = {
        "mode": config.mode,
        **asdict(config.controller) # Unpacks to lr_c=0.1 OR k_p=0.8, dt=0.1, max_steps etc.
    }
    
    controller = ControlMechanism(**controller_kwargs)
    plasticity = Plasticity(lr_w=config.plasticity.lr_w)

    # 3. Initialize variables, pbar and objects for training
    trainer = Trainer(network, controller, plasticity)
    dataloader = get_dataloader(batch_size=4, shuffle=True)

    best_loss = float('inf')
    current_avg_loss = float('inf')

    if wandb_on:
        wandb.init(
            project="target-learning", 
            name=f"{config.task}_{config.mode}_{config.dendritic_effect}_run",
            config=asdict(config) 
        )
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(range(config.epochs), desc="Learning")

    # 4. Train
    for epoch in progress_bar:
        current_avg_loss, avg_control_mag = trainer.train_one_epoch(dataloader)

        # Checkpoint: Save if this is the best model so far
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss

            if should_save: 
                save_experiment(
                    network=network, 
                    controller=controller, 
                    plasticity=plasticity, 
                    epoch=epoch, 
                    loss=current_avg_loss, 
                    task=config.task
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

    print(f"\n✅ Training Complete! Best loss {best_loss:.4f}, Final loss {current_avg_loss:.4f}")

    # Cleanly close the W&B run
    if wandb.run is not None:
        wandb.finish()

    if config.mode == "pid" and manim:

        # Grab ONE sample batch from your dataset
        test_inputs, test_targets = next(iter(dataloader))

        # Force a single control optimization pass just to harvest the data
        print("Generating Manim visualization data...")
        _, metrics = controller.optimize_control_signal(
            sensory_inputs=test_inputs,
            target_y=test_targets,
            network=network
        )

        # Save it
        history_filename = "network_history.pkl"
        with open(history_filename, "wb") as f:
            pickle.dump(metrics.state_history, f)

        print(f"Successfully exported {len(metrics.state_history)} steps to {history_filename}!")

    # Diagnostic evaluation of the final trained model on the XOR task
    with torch.no_grad():
        xor_inputs = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        targets   = torch.tensor([0., 1., 1., 0.])
        preds     = network(xor_inputs, control_signals=None, save_baseline=False).squeeze()
        predicted = (preds >= 0.5).float()

        print(f"{'input':>8} | {'target':>6} | {'raw':>8} | {'pred':>4} | {'ok':>3}")
        print("-" * 42)
        for x, t, p, pc in zip(xor_inputs, targets, preds, predicted):
            ok = "✓" if pc == t else "✗"
            print(f"{x.tolist()!s:>8} | {int(t):>6} | {p.item():>8.3f} | {int(pc):>4} | {ok:>3}")

        acc = (predicted == targets).float().mean().item()
        print(f"\naccuracy: {acc:.0%}")

if __name__ == "__main__":
    main()