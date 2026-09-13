import torch
import os
import sys
import pickle
from pathlib import Path
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd().parent  # target-learning-impl folder
for p in (root, root / "src"):
    p = str(p)
    if p not in sys.path:
        sys.path.append(p)

# ==========================================
# TURN ON ANOMALY DETECTION FOR DEBUGGING
# ==========================================
# torch.autograd.set_detect_anomaly(True)

from models.network import Network
from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from data.xor.dataset import get_dataloader as get_xor_dataloader
from data.mnist.dataset import get_dataloader as get_mnist_dataloader
from core.trainer import Trainer
from utils.utils import set_all_seeds, save_experiment, get_weight_metrics, get_logger
from utils.config import (
    ExperimentConfig,
    BackpropControlParams,
    PIDControlParams,
    BackpropPlasticityParams,
    PIDPlasticityParams,
)
from IPython.display import clear_output
from tqdm import tqdm
import wandb

logger = get_logger()


def evaluate_model(
    network, task: str, test_loader=None, verbose=True
) -> tuple[float, float]:
    """Evaluate a trained model on the selected XOR or MNIST task.

    Args:
        network (Network): Model to evaluate.
        task (str): Dataset identifier, either ``"xor"`` or ``"mnist"``.
        test_loader (DataLoader | None): MNIST loader required for MNIST evaluation.
        verbose (bool): Whether to print predictions and aggregate metrics.

    Returns:
        tuple[float, float]: Mean squared error and classification accuracy.

    Raises:
        ValueError: If the task is unsupported or the MNIST loader is missing.
    """
    network.eval()

    if task == "xor":
        inputs = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

        with torch.no_grad():
            predictions = network(
                inputs,
                control_signals=None,
                save_baseline=False,
            ).squeeze()

        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        predicted_labels = (predictions >= 0.5).float()
        accuracy = (predicted_labels == targets).float().mean().item()

        if verbose:
            print(
                f"{'input':>8} | {'target':>6} | {'raw':>8} | {'pred':>4} | {'ok':>3}"
            )
            print("-" * 42)
            for input_value, target, prediction, predicted_label in zip(
                inputs, targets, predictions, predicted_labels
            ):
                status = "ok" if predicted_label == target else "--"
                print(
                    f"{input_value.tolist()!s:>8} | {int(target):>6} | "
                    f"{prediction.item():>8.3f} | {int(predicted_label):>4} | {status:>3}"
                )
            print(f"\nXOR test loss (MSE): {mse:.6f}")
            print(f"XOR accuracy: {accuracy:.2%}")

        return mse, accuracy

    elif task == "mnist":
        if test_loader is None:
            raise ValueError("test_loader must be provided for mnist evaluation")

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = network(
                    inputs,
                    control_signals=None,
                    save_baseline=False,
                )
                batch_mse = torch.nn.functional.mse_loss(
                    predictions, targets, reduction="mean"
                )
                total_loss += batch_mse.item() * targets.size(0)

                target_labels = targets.argmax(dim=1)
                predicted_labels = predictions.argmax(dim=1)
                correct += (predicted_labels == target_labels).sum().item()
                total += targets.size(0)

        mse = total_loss / total
        accuracy = correct / total

        if verbose:
            print(f"\nMNIST test loss (MSE): {mse:.6f}")
            print(f"MNIST test accuracy: {accuracy:.2%} ({correct}/{total})")

        return mse, accuracy
    else:
        raise ValueError(f"Unsupported evaluation task: {task}")


def main():
    """Configure and run the project's default training experiment.

    Args:
        None.

    Returns:
        None.
    """
    config = ExperimentConfig(
        task="mnist",  # Choose 'mnist' or 'xor'
        mode="pid",  # Choose 'backprop' or 'pid'
        dendritic_effect="additive",  # Choose 'additive' or 'multiplicative'
        seed=42,
        controller=PIDControlParams(),
        plasticity=PIDPlasticityParams(),
    )

    set_all_seeds(config.seed)
    manim = False
    wandb_on = False
    should_save = False

    print(
        f"🚀 {config.task.upper()} with {config.mode.upper()} ({config.dendritic_effect})"
    )

    # Initialize Anatomy using config values
    network = Network(
        pop_sizes=config.pop_sizes, dendritic_effect=config.dendritic_effect
    )

    # Initialize Mechanics
    controller_kwargs = {"mode": config.mode, **asdict(config.controller)}

    controller = ControlMechanism(**controller_kwargs)
    plasticity = Plasticity(lr_w=config.plasticity.lr_w)

    # 3. Initialize variables, pbar and objects for training
    trainer = Trainer(network, controller, plasticity)

    if config.task == "mnist":
        dataloader = get_mnist_dataloader(
            batch_size=64,
            shuffle=True,
            num_classes=config.pop_sizes[-1],
        )
        test_loader = get_mnist_dataloader(
            batch_size=256,
            train=False,
            shuffle=False,
            num_workers=0,
            num_classes=config.pop_sizes[-1],
        )
    else:
        dataloader = get_xor_dataloader(batch_size=4, shuffle=True)
        test_loader = None

    best_loss = float("inf")
    current_avg_loss = float("inf")

    if wandb_on:
        wandb.init(
            project="target-learning",
            name=f"{config.task}_{config.mode}_{config.dendritic_effect}_run",
            config=asdict(config),
        )

    # Data logging containers for our plots
    epochs_list = []
    history_mse = []
    history_acc = []
    history_ctrl_mag = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(config.epochs), desc="Learning")

    # 4. Train
    for epoch in progress_bar:
        current_avg_loss, avg_control_mag = trainer.train_one_epoch(dataloader)

        # Evaluate model after each epoch without verbose prints to track metrics
        val_mse, val_acc = evaluate_model(
            network, config.task, test_loader=test_loader, verbose=False
        )

        epochs_list.append(epoch + 1)
        history_mse.append(val_mse)
        history_acc.append(val_acc)
        history_ctrl_mag.append(avg_control_mag)

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
                    task=config.task,
                )

        progress_bar.set_postfix(
            {"Loss": f"{current_avg_loss:.4f}", "Best": f"{best_loss:.4f}"}
        )

        if wandb.run is not None:
            wandb_metrics = {
                "Training Loss": current_avg_loss,
                "Control Magnitude": avg_control_mag,
                "Test MSE": val_mse,
                "Test Accuracy": val_acc,
            }
            wandb_metrics.update(get_weight_metrics(network))
            wandb.log(wandb_metrics, step=epoch)
        elif wandb_on:
            logger.warning(
                "W&B logging is enabled but no active run found. Metrics will not be logged to W&B."
            )

    print(
        f"\n✅ Training Complete! Best loss {best_loss:.4f}, Final loss {current_avg_loss:.4f}"
    )

    # Cleanly close the W&B run
    if wandb.run is not None:
        wandb.finish()

    if config.mode == "pid" and manim:
        # Grab ONE sample batch from your dataset
        test_inputs, test_targets = next(iter(dataloader))

        # Force a single control optimization pass just to harvest the data
        print("Generating Manim visualization data...")
        _, metrics = controller.optimize_control_signal(
            sensory_inputs=test_inputs, target_y=test_targets, network=network
        )

        # Save it
        history_filename = "network_history.pkl"
        with open(history_filename, "wb") as f:
            pickle.dump(metrics.state_history, f)

        print(
            f"Successfully exported {len(metrics.state_history)} steps to {history_filename}!"
        )

    # Final verbose evaluation
    evaluate_model(network, config.task, test_loader=test_loader, verbose=True)

    # ---------------------------------------------------------
    # Generate the Matplotlib Training Diagnostics Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Plot 1: Autonomous Inference MSE
    axes[0].plot(epochs_list, history_mse, label="Test MSE", color="tab:blue")
    axes[0].set_title("Autonomous Inference MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Plot 2: Accuracy
    axes[1].plot(epochs_list, history_acc, label="Test Accuracy", color="tab:orange")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    if config.task == "mnist":
        axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    # Plot 3: Control Magnitude
    axes[2].plot(
        epochs_list, history_ctrl_mag, label="Control Magnitude", color="tab:green"
    )
    axes[2].set_title("Control Magnitude")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Magnitude")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plot_filename = (
        f"{config.task}_{config.mode}_{config.dendritic_effect}_training_metrics.png"
    )
    fig.savefig(plot_filename, dpi=160)
    plt.close(fig)
    print(f"\n📊 Saved diagnostic plots to {plot_filename}")


if __name__ == "__main__":
    main()
