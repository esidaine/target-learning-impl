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
from utils.config import (
    BackpropControlParams,
    BackpropPlasticityParams,
    ExperimentConfig,
    PIDControlParams,
    PIDPlasticityParams,
)
from utils.utils import set_all_seeds


def train_one_seed(
    seed: int,
    epochs: int,
    mode: str,
    dendritic_effect: str,
    lr_w: float | None = None,
    k_p: float | None = None,
    bias_mode: str = "positive",
    refresh_feedback: bool = True,
    hidden_width: int = 8,
    feedback_mode: str = "dfc",
) -> dict[str, object]:
    """Train one XOR model and return learning and mechanism diagnostics.

    Args:
        seed (int): Random seed for initialization and data order.
        epochs (int): Number of training epochs.
        mode (str): Controller mode, typically ``"pid"`` or ``"backprop"``.
        dendritic_effect (str): Dendritic interaction mode used by the network.
        lr_w (float | None): Optional override for the plasticity learning rate.
        k_p (float | None): Optional override for the PID proportional gain.
        bias_mode (str): Bias initialization strategy.
        refresh_feedback (bool): Whether to refresh feedback weights after updates.
        hidden_width (int): Width of the hidden layer.
        feedback_mode (str): Feedback projection mode used by the controller.

    Returns:
        dict[str, object]: Per-seed training curves and final evaluation metrics.
    """
    set_all_seeds(seed)

    if mode == "pid":
        controller_config = PIDControlParams()
        plasticity_config = PIDPlasticityParams()
    else:
        controller_config = BackpropControlParams()
        plasticity_config = BackpropPlasticityParams()

    if lr_w is not None:
        plasticity_config.lr_w = lr_w
    if k_p is not None and mode == "pid":
        controller_config.k_p = k_p

    config = ExperimentConfig(
        task="xor",
        mode=mode,
        seed=seed,
        epochs=epochs,
        dendritic_effect=dendritic_effect,
        controller=controller_config,
        plasticity=plasticity_config,
    )
    config.pop_sizes = [2, hidden_width, 1]

    network = Network(
        pop_sizes=config.pop_sizes,
        dendritic_effect=config.dendritic_effect,
    )
    with torch.no_grad():
        for population in network.populations:
            if bias_mode == "random":
                population.W.bias.uniform_(-0.5, 0.5)
            elif bias_mode == "zero":
                population.W.bias.zero_()
            else:
                population.W.bias.fill_(0.1)
    controller = ControlMechanism(
        mode=mode,
        feedback_mode=feedback_mode,
        **vars(config.controller),
    )
    plasticity = Plasticity(lr_w=config.plasticity.lr_w)
    dataloader = get_dataloader(batch_size=4, shuffle=True)

    epoch_losses: list[float] = []
    control_failure_rates: list[float] = []
    control_magnitudes: list[float] = []
    unstable = False

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_control_failures = 0
        epoch_control_magnitude = 0.0

        for sensory_inputs, target_y in dataloader:
            controls, metrics = controller.optimize_control_signal(
                sensory_inputs=sensory_inputs,
                target_y=target_y,
                network=network,
            )
            plasticity.update_weights(network=network, sensory_inputs=sensory_inputs)
            if refresh_feedback:
                network.refresh_feedback_weights()

            baseline_predictions = network.populations[-1].a_baseline
            if baseline_predictions is None:
                raise RuntimeError("Baseline predictions were not recorded.")

            batch_loss = torch.nn.functional.mse_loss(baseline_predictions, target_y)
            epoch_loss += batch_loss.item()
            epoch_control_failures += int(not metrics.improved)
            epoch_control_magnitude += sum(
                control.abs().mean().item() for control in controls
            ) / len(controls)

            tensors = [batch_loss, *controls]
            if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
                unstable = True

        n_batches = len(dataloader)
        epoch_loss /= n_batches
        epoch_control_failure_rate = epoch_control_failures / n_batches
        epoch_control_magnitude /= n_batches
        epoch_losses.append(epoch_loss)
        control_failure_rates.append(epoch_control_failure_rate)
        control_magnitudes.append(epoch_control_magnitude)
        if not np.isfinite(epoch_loss):
            unstable = True

        print(
            f"{mode:>8} {dendritic_effect:>14} seed={seed:>3} "
            f"epoch={epoch + 1:>4}/{epochs} baseline_mse={epoch_loss:.6f} "
            f"control_mag={epoch_control_magnitude:.6f}"
        )

    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    network.eval()
    with torch.no_grad():
        final_predictions = network(
            inputs, control_signals=None, save_baseline=False
        ).reshape(-1, 1)

    final_mse = torch.nn.functional.mse_loss(final_predictions, targets).item()
    final_labels = (final_predictions >= 0.5).float()
    final_accuracy = (final_labels == targets).float().mean().item()
    unstable = unstable or not torch.isfinite(final_predictions).all().item()
    unstable = unstable or any(
        not torch.isfinite(pop.W.weight).all().item()
        or not torch.isfinite(pop.W.bias).all().item()
        for pop in network.populations
    )

    return {
        "seed": seed,
        "hidden_width": hidden_width,
        "feedback_mode": feedback_mode,
        "epoch_losses": epoch_losses,
        "control_failure_rates": control_failure_rates,
        "control_magnitudes": control_magnitudes,
        "final_no_control_mse": final_mse,
        "final_xor_accuracy": final_accuracy,
        "raw_predictions": [float(value) for value in final_predictions.squeeze(1)],
        "unstable": unstable,
    }


def plot_learning_curve(
    histories: dict[int, list[float]],
    output_path: Path,
    mode: str,
    dendritic_effect: str,
) -> None:
    """Save individual seed traces and the mean +/- one standard deviation.

    Args:
        histories (dict[int, list[float]]): Mapping from seed to per-epoch loss curve.
        output_path (Path): Destination path for the rendered plot.
        mode (str): Controller mode used in the experiment.
        dendritic_effect (str): Dendritic effect used in the experiment.

    Returns:
        None.
    """
    values = np.asarray(list(histories.values()), dtype=float)
    epochs = np.arange(1, values.shape[1] + 1)
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    figure, axis = plt.subplots(figsize=(8, 5))
    for seed, losses in histories.items():
        axis.plot(epochs, losses, color="0.72", linewidth=1.0, alpha=0.8)

    axis.plot(
        epochs,
        mean,
        color="#1f4e79",
        linewidth=2.4,
        label="Mean across seeds",
    )
    axis.fill_between(
        epochs,
        mean - std,
        mean + std,
        color="#6baed6",
        alpha=0.25,
        label="Mean +/- 1 SD",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Autonomous Inference MSE")
    axis.set_title(f"XOR learning curve: {mode}, {dendritic_effect} dendritic effect")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_seed_diagnostics(
    results: list[dict[str, object]],
    output_path: Path,
    mse_threshold: float,
) -> None:
    """Plot training diagnostics as seed-wise lines over epochs.

    Final MSE, accuracy, threshold epoch, and instability remain in the JSON
    and CSV summaries because they are single end-of-run values, not curves.

    Args:
        results (list[dict[str, object]]): Per-seed experiment results.
        output_path (Path): Destination path for the diagnostics plot.
        mse_threshold (float): MSE threshold line shown in the first subplot.

    Returns:
        None.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_specs = [
        ("epoch_losses", "Autonomous inference MSE", "MSE", True),
        ("control_failure_rates", "Control failure rate", "Failure rate", False),
        ("control_magnitudes", "Control magnitude", "Mean |control|", False),
    ]

    for axis, (key, title, ylabel, log_scale) in zip(axes, plot_specs):
        for result in results:
            values = np.asarray(result[key], dtype=float)
            epochs = np.arange(1, len(values) + 1)
            if key == "control_failure_rates":
                values = values * 100.0
                axis_ylabel = "Failure rate (%)"
            else:
                axis_ylabel = ylabel
            axis.plot(
                epochs,
                values,
                linewidth=1.5,
                label=f"Seed {result['seed']}",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(axis_ylabel)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)

    axes[0].axhline(
        mse_threshold,
        color="0.4",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold ({mse_threshold:g})",
    )
    axes[0].legend(frameon=False, fontsize=8)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_final_predictions(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot the four final autonomous-inference XOR predictions for every seed.

    Args:
        results (list[dict[str, object]]): Per-seed experiment results.
        output_path (Path): Destination path for the predictions plot.

    Returns:
        None.
    """
    labels = ["00", "01", "10", "11"]
    targets = np.array([0.0, 1.0, 1.0, 0.0])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(labels, targets, "ko-", linewidth=2, label="Target")
    for result in results:
        axis.plot(
            labels,
            result["raw_predictions"],
            "o--",
            linewidth=1.2,
            alpha=0.75,
            label=f"Seed {result['seed']}",
        )
    axis.axhline(0.5, color="0.65", linestyle=":", label="Classification threshold")
    axis.set_xlabel("XOR input")
    axis.set_ylabel("Final autonomous inference output")
    axis.set_title("Final XOR predictions by seed")
    axis.set_ylim(-0.1, 1.1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the XOR mechanism experiment.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed training, ablation, and output options.
    """
    parser = argparse.ArgumentParser(
        description="Plot autonomous inference XOR MSE across multiple random seeds."
    )
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 42, 27, 37, 47, 17, 23, 31, 53, 71],
    )
    parser.add_argument("--mode", choices=["pid", "backprop"], default="pid")
    parser.add_argument("--lr-w", type=float, default=None)
    parser.add_argument("--k-p", type=float, default=None)
    parser.add_argument("--hidden-width", type=int, default=8)
    parser.add_argument("--feedback-mode", choices=["dfc", "chain"], default="dfc")
    feedback_group = parser.add_mutually_exclusive_group()
    feedback_group.add_argument(
        "--refresh-feedback",
        dest="refresh_feedback",
        action="store_true",
        help="Refresh Q after each plasticity update (default).",
    )
    feedback_group.add_argument(
        "--frozen-feedback",
        dest="refresh_feedback",
        action="store_false",
        help="Keep Q fixed at initialization for the ablation condition.",
    )
    parser.set_defaults(refresh_feedback=True)
    parser.add_argument(
        "--bias-mode",
        choices=["random", "zero", "positive"],
        default="positive",
        help="Bias initialization: +0.1 by default, random range, or zero.",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.01,
        help="MSE threshold used to report the first successful epoch.",
    )
    parser.add_argument(
        "--dendritic-effect",
        choices=["additive", "multiplicative"],
        default="additive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation_results" / "xor_no_control_mse.png",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=ROOT / "evaluation_results" / "xor_mechanism_diagnostics.png",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=ROOT / "evaluation_results" / "xor_final_predictions.png",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "evaluation_results" / "xor_mechanism_results.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the XOR mechanism experiment and write diagnostic artifacts.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    results = [
        train_one_seed(
            seed=seed,
            epochs=args.epochs,
            mode=args.mode,
            dendritic_effect=args.dendritic_effect,
            lr_w=args.lr_w,
            k_p=args.k_p,
            refresh_feedback=args.refresh_feedback,
            bias_mode=args.bias_mode,
            hidden_width=args.hidden_width,
            feedback_mode=args.feedback_mode,
        )
        for seed in args.seeds
    ]

    for result in results:
        result["threshold_epoch"] = next(
            (
                epoch
                for epoch, loss in enumerate(result["epoch_losses"], start=1)
                if loss <= args.mse_threshold
            ),
            None,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(
        histories={int(result["seed"]): result["epoch_losses"] for result in results},
        output_path=args.output,
        mode=args.mode,
        dendritic_effect=args.dendritic_effect,
    )
    plot_seed_diagnostics(results, args.diagnostics_output, args.mse_threshold)
    plot_final_predictions(results, args.predictions_output)

    args.results_output.write_text(
        json.dumps(
            {
                "config": {
                    "epochs": args.epochs,
                    "seeds": args.seeds,
                    "mode": args.mode,
                    "dendritic_effect": args.dendritic_effect,
                    "lr_w": args.lr_w,
                    "k_p": args.k_p,
                    "bias_mode": args.bias_mode,
                    "refresh_feedback": args.refresh_feedback,
                    "hidden_width": args.hidden_width,
                    "feedback_mode": args.feedback_mode,
                    "mse_threshold": args.mse_threshold,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = args.results_output.with_suffix(".csv")
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "final_no_control_mse",
                "final_xor_accuracy",
                "threshold_epoch",
                "unstable",
                "mean_control_failure_rate",
                "mean_control_magnitude",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "final_no_control_mse": result["final_no_control_mse"],
                    "final_xor_accuracy": result["final_xor_accuracy"],
                    "threshold_epoch": result["threshold_epoch"],
                    "unstable": result["unstable"],
                    "mean_control_failure_rate": np.mean(
                        result["control_failure_rates"]
                    ),
                    "mean_control_magnitude": np.mean(result["control_magnitudes"]),
                }
            )

    print(f"Saved plot to {args.output}")
    print(f"Saved diagnostics plot to {args.diagnostics_output}")
    print(f"Saved predictions plot to {args.predictions_output}")
    print(f"Saved per-seed results to {args.results_output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
