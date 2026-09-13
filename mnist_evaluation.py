"""Run matched MNIST evaluations and plot additive/multiplicative results.

Example:
    python mnist_evaluation.py --epochs 10 --seeds 7 17 27 37 47

The evaluation keeps the test split out of training and writes one JSON file,
one CSV file, and several PNG plots into the configured output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from data.mnist.dataset import get_dataloader
from models.network import Network
from utils.config import ExperimentConfig, PIDControlParams, PIDPlasticityParams
from utils.utils import set_all_seeds


def evaluate_test_set(
    network: Network, loader: torch.utils.data.DataLoader
) -> dict[str, float]:
    """Return held-out MSE and classification accuracy for one model.

    Args:
        network (Network): Model to evaluate.
        loader (torch.utils.data.DataLoader): Held-out data loader.

    Returns:
        dict[str, float]: Dictionary with ``test_mse`` and ``test_accuracy``.
    """
    network.eval()
    total_squared_error = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = network(inputs, control_signals=None, save_baseline=False)
            batch_mse = torch.nn.functional.mse_loss(
                predictions, targets, reduction="mean"
            )
            total_squared_error += batch_mse.item() * targets.size(0)
            correct += (predictions.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            total += targets.size(0)
    return {
        "test_mse": total_squared_error / total,
        "test_accuracy": correct / total,
    }


def run_experiment(
    effect: str,
    seed: int,
    epochs: int,
    batch_size: int,
    data_root: Path,
) -> dict[str, Any]:
    """Train one configuration and return epoch-level diagnostics.

    Args:
        effect (str): Dendritic effect to evaluate.
        seed (int): Random seed for reproducible initialization and data order.
        epochs (int): Number of training epochs to run.
        batch_size (int): Mini-batch size for the training loader.
        data_root (Path): Root directory containing MNIST data.

    Returns:
        dict[str, Any]: Per-run metadata and epoch-level metrics.
    """
    config = ExperimentConfig(
        task="mnist",
        mode="pid",
        dendritic_effect=effect,
        seed=seed,
        epochs=epochs,
        controller=PIDControlParams(),
        plasticity=PIDPlasticityParams(),
    )
    set_all_seeds(seed)
    network = Network(pop_sizes=config.pop_sizes, dendritic_effect=effect)
    controller = ControlMechanism(mode=config.mode, **asdict(config.controller))
    plasticity = Plasticity(lr_w=config.plasticity.lr_w)
    train_loader = get_dataloader(
        batch_size=batch_size,
        root_dir=str(data_root),
        train=True,
        shuffle=True,
        num_classes=config.pop_sizes[-1],
    )
    test_loader = get_dataloader(
        batch_size=256,
        root_dir=str(data_root),
        train=False,
        shuffle=False,
        num_classes=config.pop_sizes[-1],
    )

    history: list[dict[str, float | int | bool]] = []
    for epoch in range(epochs):
        network.train()
        train_loss = 0.0
        control_magnitude = 0.0
        control_improvement = 0.0
        control_failures = 0
        output_mse_before_control = 0.0
        output_mse_after_control = 0.0
        layer_control_magnitude = [0.0] * len(network.populations)
        layer_control_improvement = [0.0] * len(network.populations)
        layer_inactive_fraction = [0.0] * len(network.populations)
        finite = True

        for inputs, targets in train_loader:
            controls, metrics = controller.optimize_control_signal(
                sensory_inputs=inputs, target_y=targets, network=network
            )
            if not metrics.improved:
                control_failures += 1
            control_improvement += metrics.improvement
            control_magnitude += sum(c.abs().mean().item() for c in controls) / len(
                controls
            )
            output_mse_before_control += metrics.initial_loss
            output_mse_after_control += metrics.final_loss
            for layer_index, (pop, control) in enumerate(
                zip(network.populations, controls)
            ):
                baseline = pop.a_baseline
                controlled = pop.a_controlled
                target_activation = pop.target_activation
                layer_control_magnitude[layer_index] += control.abs().mean().item()
                if (
                    baseline is not None
                    and controlled is not None
                    and target_activation is not None
                ):
                    baseline_error = torch.nn.functional.mse_loss(
                        baseline, target_activation
                    )
                    controlled_error = torch.nn.functional.mse_loss(
                        controlled, target_activation
                    )
                    layer_control_improvement[layer_index] += (
                        baseline_error.item() - controlled_error.item()
                    )
                if pop.z is not None:
                    layer_inactive_fraction[layer_index] += (
                        (pop.z <= 0).float().mean().item()
                    )
            plasticity.update_weights(network=network, sensory_inputs=inputs)
            network.refresh_feedback_weights()

            baseline = network.populations[-1].a_baseline
            batch_loss = torch.nn.functional.mse_loss(baseline, targets)
            train_loss += batch_loss.item()
            finite = finite and bool(torch.isfinite(batch_loss))

        test_metrics = evaluate_test_set(network, test_loader)
        weight_norm = max(
            pop.W.weight.detach().norm().item() for pop in network.populations
        )
        finite = finite and all(
            torch.isfinite(torch.tensor(value)) for value in test_metrics.values()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": train_loss / len(train_loader),
                "test_mse": test_metrics["test_mse"],
                "test_accuracy": test_metrics["test_accuracy"],
                "control_magnitude": control_magnitude / len(train_loader),
                "control_improvement": control_improvement / len(train_loader),
                "output_mse_before_control": output_mse_before_control
                / len(train_loader),
                "output_mse_after_control": output_mse_after_control
                / len(train_loader),
                "control_failures": control_failures,
                "control_failure_rate": control_failures / len(train_loader),
                "max_weight_norm": weight_norm,
                "layer_weight_norms": [
                    pop.W.weight.detach().norm().item() for pop in network.populations
                ],
                "layer_control_magnitudes": [
                    value / len(train_loader) for value in layer_control_magnitude
                ],
                "layer_control_improvements": [
                    value / len(train_loader) for value in layer_control_improvement
                ],
                "layer_inactive_fractions": [
                    value / len(train_loader) for value in layer_inactive_fraction
                ],
                "finite": finite,
            }
        )
        print(
            f"{effect:>14} seed={seed:>3} epoch={epoch + 1:>2}/{epochs} "
            f"accuracy={test_metrics['test_accuracy']:.2%} "
            f"test_mse={test_metrics['test_mse']:.5f}"
        )

    return {
        "effect": effect,
        "seed": seed,
        "config": asdict(config),
        "history": history,
    }


def save_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Write evaluation records, tables, and comparison plots.

    Args:
        results (list[dict[str, Any]]): Experiment records with epoch histories.
        output_dir (Path): Directory receiving JSON, CSV, and PNG outputs.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnist_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with (output_dir / "mnist_epoch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        rows = [
            {"effect": run["effect"], "seed": run["seed"], **epoch}
            for run in results
            for epoch in run["history"]
        ]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_learning_curves(results, output_dir / "mnist_learning_curves.png")
    plot_final_accuracy(results, output_dir / "mnist_final_accuracy.png")
    plot_diagnostics(results, output_dir)
    plot_layer_diagnostics(results, output_dir / "mnist_layer_diagnostics.png")


def plot_learning_curves(results: list[dict[str, Any]], path: Path) -> None:
    """Plot mean +/- one standard deviation over seeds.

    Args:
        results (list[dict[str, Any]]): Experiment records grouped by effect and seed.
        path (Path): Output path for the saved figure.

    Returns:
        None.
    """
    effects = sorted({run["effect"] for run in results})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("test_accuracy", "Test accuracy", "Accuracy"),
        ("test_mse", "Test MSE", "MSE"),
        ("train_mse", "Train MSE", "MSE"),
        ("control_failure_rate", "Control failure rate", "Rate"),
    ]
    for axis, (key, title, ylabel) in zip(axes.flat, metrics):
        for effect in effects:
            curves = [
                [epoch[key] for epoch in run["history"]]
                for run in results
                if run["effect"] == effect
            ]
            values = torch.tensor(curves, dtype=torch.float64)
            epochs = range(1, values.shape[1] + 1)
            mean = values.mean(dim=0).numpy()
            std = values.std(dim=0, unbiased=False).numpy()
            axis.plot(epochs, mean, label=effect)
            axis.fill_between(epochs, mean - std, mean + std, alpha=0.16)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("MNIST target-learning evaluation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_final_accuracy(results: list[dict[str, Any]], path: Path) -> None:
    """Plot final held-out accuracy for each seed and effect.

    Args:
        results (list[dict[str, Any]]): Experiment records grouped by effect and seed.
        path (Path): Output path for the saved figure.

    Returns:
        None.
    """
    effects = sorted({run["effect"] for run in results})
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = torch.arange(len(effects), dtype=torch.float64).numpy()
    for index, effect in enumerate(effects):
        values = [
            run["history"][-1]["test_accuracy"]
            for run in results
            if run["effect"] == effect
        ]
        x = (
            positions[index]
            + (
                torch.arange(len(values), dtype=torch.float64).numpy()
                - (len(values) - 1) / 2
            )
            * 0.04
        )
        axis.scatter(x, values, alpha=0.8, label=f"{effect} seeds")
        axis.errorbar(
            index,
            sum(values) / len(values),
            yerr=torch.tensor(values).std(unbiased=False).item(),
            fmt="_",
            color="black",
            capsize=5,
            linewidth=2,
        )
    axis.set_xticks(positions, effects)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Final test accuracy")
    axis.set_title("Final MNIST accuracy by dendritic effect")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diagnostics(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot weight growth and controller behavior over training.

    Args:
        results (list[dict[str, Any]]): Experiment records grouped by effect and seed.
        output_dir (Path): Directory receiving generated diagnostic plots.

    Returns:
        None.
    """
    effects = sorted({run["effect"] for run in results})
    metrics = [
        (
            "max_weight_norm",
            "Maximum forward-weight norm",
            "Norm",
            "mnist_max_weight_norm.png",
        ),
        (
            "control_magnitude",
            "Mean control magnitude",
            "Magnitude",
            "mnist_control_magnitude.png",
        ),
        (
            "control_failure_rate",
            "Control failure rate",
            "Failure rate",
            "mnist_control_failure_rate.png",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (key, title, ylabel, filename) in zip(axes, metrics):
        for effect in effects:
            curves = [
                [epoch[key] for epoch in run["history"]]
                for run in results
                if run["effect"] == effect
            ]
            values = torch.tensor(curves, dtype=torch.float64)
            epochs = range(1, values.shape[1] + 1)
            mean = values.mean(dim=0).numpy()
            std = values.std(dim=0, unbiased=False).numpy()
            axis.plot(epochs, mean, label=effect)
            axis.fill_between(epochs, mean - std, mean + std, alpha=0.16)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig_single, axis_single = plt.subplots(
            figsize=(6, 4.5), constrained_layout=True
        )
        for effect in effects:
            curves = [
                [epoch[key] for epoch in run["history"]]
                for run in results
                if run["effect"] == effect
            ]
            values = torch.tensor(curves, dtype=torch.float64)
            epochs = range(1, values.shape[1] + 1)
            mean = values.mean(dim=0).numpy()
            std = values.std(dim=0, unbiased=False).numpy()
            axis_single.plot(epochs, mean, label=effect)
            axis_single.fill_between(epochs, mean - std, mean + std, alpha=0.16)
        axis_single.set_title(title)
        axis_single.set_xlabel("Epoch")
        axis_single.set_ylabel(ylabel)
        axis_single.grid(alpha=0.25)
        axis_single.legend()
        fig_single.savefig(output_dir / filename, dpi=160)
        plt.close(fig_single)

    fig.suptitle("MNIST training diagnostics")
    fig.savefig(output_dir / "mnist_training_diagnostics.png", dpi=160)
    plt.close(fig)


def plot_layer_diagnostics(results: list[dict[str, Any]], path: Path) -> None:
    """Plot per-layer diagnostics and output error before/after control.

    Args:
        results (list[dict[str, Any]]): Experiment records grouped by effect and seed.
        path (Path): Output path for the saved figure.

    Returns:
        None.
    """
    effects = sorted({run["effect"] for run in results})
    layer_count = len(results[0]["history"][0]["layer_weight_norms"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    metrics = [
        ("layer_weight_norms", "Weight norm", "Norm", False),
        ("layer_control_magnitudes", "Control magnitude", "Magnitude", False),
        ("layer_inactive_fractions", "Inactive ReLU fraction", "Fraction", True),
        (
            "layer_control_improvements",
            "Local control improvement",
            "Improvement",
            False,
        ),
    ]
    for axis, (key, title, ylabel, percent) in zip(axes.flat, metrics):
        for effect in effects:
            for layer_index in range(layer_count):
                curves = [
                    [epoch[key][layer_index] for epoch in run["history"]]
                    for run in results
                    if run["effect"] == effect
                ]
                values = torch.tensor(curves, dtype=torch.float64)
                epochs = range(1, values.shape[1] + 1)
                mean = values.mean(dim=0).numpy()
                label = f"{effect} L{layer_index + 1}"
                axis.plot(epochs, mean, label=label, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if percent:
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)

    axis = axes[1, 2]
    for effect in effects:
        before = []
        after = []
        for run in results:
            if run["effect"] == effect:
                before.append(
                    [epoch["output_mse_before_control"] for epoch in run["history"]]
                )
                after.append(
                    [epoch["output_mse_after_control"] for epoch in run["history"]]
                )
        before_values = torch.tensor(before, dtype=torch.float64).mean(dim=0).numpy()
        after_values = torch.tensor(after, dtype=torch.float64).mean(dim=0).numpy()
        epochs = range(1, len(before_values) + 1)
        axis.plot(epochs, before_values, label=f"{effect} before")
        axis.plot(epochs, after_values, linestyle="--", label=f"{effect} after")
    axis.set_title("Output MSE before and after control")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.suptitle("MNIST per-layer and control diagnostics")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the MNIST evaluation script.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed evaluation and output configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mnist_evaluation")
    return parser.parse_args()


def main() -> None:
    """Run additive and multiplicative MNIST evaluations for each seed.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    results = [
        run_experiment(effect, seed, args.epochs, args.batch_size, args.data_root)
        for effect in ("additive", "multiplicative")
        for seed in args.seeds
    ]
    save_results(results, args.output_dir)
    print(f"Saved evaluation data and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
