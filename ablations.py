"""Run report-ready XOR mechanism ablations.

The default experiment compares three conditions under multiplicative
 dendritic modulation:

1. Full method: control, local plasticity, and feedback refresh.
2. No control: no controller and no weight update.
3. Controller only: control is applied, but local plasticity is disabled.

Example:
    .venv/Scripts/python.exe ablations.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.controllers import ControlMechanism
from core.plasticity import Plasticity
from data.xor.dataset import get_dataloader
from models.network import Network
from utils.config import PIDControlParams, PIDPlasticityParams
from utils.utils import set_all_seeds

CONDITIONS = ("full_method", "no_control", "controller_only")
CONDITION_LABELS = {
    "full_method": "Full method",
    "no_control": "No control",
    "controller_only": "Controller only",
}


def xor_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the four XOR inputs and their binary targets.

    Args:
        None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Input tensor of shape ``(4, 2)`` and
            target tensor of shape ``(4, 1)``.
    """
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return inputs, targets


def make_network(hidden_width: int) -> Network:
    """Construct a multiplicative-dendritic XOR network.

    Args:
        hidden_width (int): Number of neurons in the hidden population.

    Returns:
        Network: Initialized network with positive biases.
    """
    network = Network(
        pop_sizes=[2, hidden_width, 1],
        dendritic_effect="multiplicative",
    )
    with torch.no_grad():
        for population in network.populations:
            population.W.bias.fill_(0.1)
    return network


def evaluate(network: Network) -> tuple[float, float]:
    """Evaluate autonomous XOR predictions with MSE and accuracy.

    Args:
        network (Network): Network to evaluate.

    Returns:
        tuple[float, float]: Mean squared error and thresholded accuracy.
    """
    inputs, targets = xor_batch()
    network.eval()
    with torch.no_grad():
        predictions = network(inputs, control_signals=None, save_baseline=False)
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    accuracy = ((predictions >= 0.5) == targets).float().mean().item()
    return mse, accuracy


def train_condition(
    condition: str,
    seed: int,
    epochs: int,
    hidden_width: int,
) -> dict[str, object]:
    """Train one ablation condition and collect epoch-level metrics.

    Args:
        condition (str): Ablation condition to run.
        seed (int): Random seed for the experiment.
        epochs (int): Number of training epochs.
        hidden_width (int): Number of hidden neurons.

    Returns:
        dict[str, object]: Training histories and final evaluation metrics.
    """
    set_all_seeds(seed)
    network = make_network(hidden_width)
    controller = ControlMechanism(
        mode="pid",
        feedback_mode="dfc",
        **vars(PIDControlParams(max_steps=100)),
    )
    plasticity = Plasticity(lr_w=0.05)
    inputs, targets = xor_batch()
    dataloader = get_dataloader(batch_size=4, shuffle=True)

    autonomous_mse: list[float] = []
    autonomous_accuracy: list[float] = []
    controlled_mse: list[float] = []
    control_magnitude: list[float] = []
    control_failure_rate: list[float] = []

    for _ in range(epochs):
        if condition == "no_control":
            mse, accuracy = evaluate(network)
            autonomous_mse.append(mse)
            autonomous_accuracy.append(accuracy)
            controlled_mse.append(mse)
            control_magnitude.append(0.0)
            control_failure_rate.append(0.0)
            continue

        batch_controlled_mse = []
        batch_control_magnitude = []
        failures = 0
        for sensory_inputs, target_y in dataloader:
            controls, metrics = controller.optimize_control_signal(
                sensory_inputs=sensory_inputs,
                target_y=target_y,
                network=network,
            )
            batch_controlled_mse.append(float(metrics.final_loss))
            batch_control_magnitude.append(
                sum(control.abs().mean().item() for control in controls) / len(controls)
            )
            failures += int(not metrics.improved)

            if condition == "full_method":
                plasticity.update_weights(
                    network=network, sensory_inputs=sensory_inputs
                )
                network.refresh_feedback_weights()

        mse, accuracy = evaluate(network)
        autonomous_mse.append(mse)
        autonomous_accuracy.append(accuracy)
        controlled_mse.append(float(np.mean(batch_controlled_mse)))
        control_magnitude.append(float(np.mean(batch_control_magnitude)))
        control_failure_rate.append(failures / len(dataloader))

    final_mse, final_accuracy = evaluate(network)
    return {
        "condition": condition,
        "seed": seed,
        "hidden_width": hidden_width,
        "dendritic_effect": "multiplicative",
        "bias": 0.1,
        "feedback_refresh": condition == "full_method",
        "autonomous_mse": autonomous_mse,
        "autonomous_accuracy": autonomous_accuracy,
        "controlled_mse": controlled_mse,
        "control_magnitude": control_magnitude,
        "control_failure_rate": control_failure_rate,
        "final_autonomous_mse": final_mse,
        "final_autonomous_accuracy": final_accuracy,
        "raw_predictions": [float(value) for value in evaluate_predictions(network)],
    }


def evaluate_predictions(network: Network) -> torch.Tensor:
    """Return autonomous predictions for the four XOR inputs.

    Args:
        network (Network): Network to evaluate.

    Returns:
        torch.Tensor: Flattened predictions ordered as ``00, 01, 10, 11``.
    """
    inputs, _ = xor_batch()
    network.eval()
    with torch.no_grad():
        return network(inputs, control_signals=None, save_baseline=False).reshape(-1)


def summarize(
    results: list[dict[str, object]], threshold: float
) -> list[dict[str, object]]:
    """Aggregate final MSE and accuracy by ablation condition.

    Args:
        results (list[dict[str, object]]): Per-seed experiment records.
        threshold (float): MSE threshold counted as a successful result.

    Returns:
        list[dict[str, object]]: One summary record for each condition.
    """
    summaries = []
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        final_mse = np.asarray([row["final_autonomous_mse"] for row in rows])
        final_accuracy = np.asarray([row["final_autonomous_accuracy"] for row in rows])
        summaries.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "mean_final_mse": float(final_mse.mean()),
                "mse_std": float(final_mse.std()),
                "threshold_success_percent": float(
                    (final_mse < threshold).mean() * 100
                ),
                "mean_final_accuracy_percent": float(final_accuracy.mean() * 100),
            }
        )
    return summaries


def plot_results(
    results: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_path: Path,
    threshold: float,
) -> None:
    """Plot autonomous learning, controlled error, and final ablation metrics.

    Args:
        results (list[dict[str, object]]): Per-seed experiment records.
        summaries (list[dict[str, object]]): Aggregated condition summaries.
        output_path (Path): Destination image path.
        threshold (float): MSE threshold shown on the learning plot.

    Returns:
        None.
    """
    colors = {
        "full_method": "#176b87",
        "no_control": "#9a9a9a",
        "controller_only": "#d97925",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        curves = np.asarray([row["autonomous_mse"] for row in rows], dtype=float)
        epochs = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        axes[0].plot(
            epochs, mean, color=colors[condition], label=CONDITION_LABELS[condition]
        )
        axes[0].fill_between(
            epochs,
            np.maximum(mean - std, 1e-8),
            mean + std,
            color=colors[condition],
            alpha=0.12,
        )

    axes[0].axhline(
        threshold, color="0.35", linestyle=":", label=f"MSE = {threshold:g}"
    )
    axes[0].set_title("Autonomous learning")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Autonomous inference MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    controller_rows = [row for row in results if row["condition"] == "controller_only"]
    controlled = np.asarray(
        [row["controlled_mse"] for row in controller_rows], dtype=float
    ).mean(axis=0)
    autonomous = np.asarray(
        [row["autonomous_mse"] for row in controller_rows], dtype=float
    ).mean(axis=0)
    epochs = np.arange(1, len(controlled) + 1)
    axes[1].plot(epochs, autonomous, color="#176b87", label="Autonomous MSE")
    axes[1].plot(
        epochs, controlled, color="#d97925", linestyle="--", label="Controlled MSE"
    )
    axes[1].set_title("Controller-only condition")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    labels = [summary["label"] for summary in summaries]
    x = np.arange(len(labels))
    final_mse = [summary["mean_final_mse"] for summary in summaries]
    success = [summary["threshold_success_percent"] for summary in summaries]
    axes[2].bar(
        x,
        np.maximum(final_mse, 1e-8),
        color=[colors[condition] for condition in CONDITIONS],
    )
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_title("Final autonomous MSE")
    axes[2].set_ylabel("Mean MSE")
    axes[2].grid(axis="y", alpha=0.25)
    for index, value in enumerate(final_mse):
        axes[2].text(index, value * 1.2, f"{value:.3f}", ha="center", fontsize=8)

    figure.suptitle(
        "XOR mechanism ablation: multiplicative dendritic modulation", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_outputs(
    results: list[dict[str, object]],
    summaries: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Write ablation records and a publication-ready summary table.

    Args:
        results (list[dict[str, object]]): Per-seed experiment records.
        summaries (list[dict[str, object]]): Aggregated condition summaries.
        output_dir (Path): Directory receiving JSON, CSV, and LaTeX outputs.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xor_ablation_results.json").write_text(
        json.dumps({"results": results, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "xor_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "xor_ablation_summary.tex").open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[htbp]\n\\centering\n")
        file.write(
            "\\caption{XOR mechanism ablation under multiplicative dendritic modulation.}\n"
        )
        file.write("\\label{tab:xor_mechanism_ablation}\n")
        file.write("\\begin{tabular}{lccc}\n\\hline\n")
        file.write(
            "Condition & Mean autonomous MSE & Seeds with MSE $<0.01$ & Mean accuracy \\\\\n"
        )
        file.write("\\hline\n")
        for summary in summaries:
            file.write(
                f"{summary['label']} & {summary['mean_final_mse']:.4f} & "
                f"{summary['threshold_success_percent']:.0f}\\% & "
                f"{summary['mean_final_accuracy_percent']:.0f}\\% \\\\\n"
            )
        file.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    """Run the configured XOR ablation matrix and save its reports.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[7, 42, 27, 37, 47, 17, 23, 31, 53, 71]
    )
    parser.add_argument("--hidden-width", type=int, default=8)
    parser.add_argument("--mse-threshold", type=float, default=0.01)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "evaluation_results" / "xor_ablations"
    )
    args = parser.parse_args()

    results = [
        train_condition(condition, seed, args.epochs, args.hidden_width)
        for condition in CONDITIONS
        for seed in args.seeds
    ]
    summaries = summarize(results, args.mse_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        results,
        summaries,
        args.output_dir / "xor_ablation_comparison.png",
        args.mse_threshold,
    )
    save_outputs(results, summaries, args.output_dir)
    print(f"Saved XOR ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
