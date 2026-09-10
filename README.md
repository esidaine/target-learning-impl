# Target Learning with Dynamical Feedback Control

An experimental implementation of local target-learning rules with dynamical
feedback control for multilayer neural networks.

## Introduction

In ordinary feedforward learning, an input is passed through a network and the
weights are updated from an output error. This project studies a different
training mechanism: a top-down controller temporarily pushes the network's
neurons toward a desired output, and local synaptic plasticity records the
resulting activity difference. The aim is for the network to produce similar
internal activity from the input alone on later presentations.

More precisely, this repository is a research codebase for studying how
top-down control can create learning targets for hidden neurons, followed by
local synaptic updates that make those targets achievable without control on
future presentations. The implementation is inspired by Target Learning and
Deep Feedback Control (DFC), but it is not a reproduction of either paper.


## Scientific Position

The project combines three ideas:

- **Target Learning:** neurons are driven toward desired activity states, and
  plasticity reduces the difference between controlled and baseline activity.
- **Deep Feedback Control:** a dynamical controller broadcasts output error
  through feedback matrices to influence populations during settling.
- **Dendritic modulation:** the code supports additive and
  multiplicative combinations of bottom-up and top-down signals.

## Method

For population $i$, let $W_i$ denote forward weights and $Q_i$ feedback
weights. Each minibatch passes through three phases.

### Baseline phase

The input is propagated without top-down control:

$$
a_i^{\mathrm{base}} = f_i(W_i a_{i-1}^{\mathrm{base}}).
$$

Baseline activity is stored for the plasticity phase.

### Control phase

The controller computes a global output control signal $u$ from the difference
between the desired and current output. Local controls are obtained through
feedback projection:

$$
c_i = Q_i u.
$$

The current implementation uses a leaky proportional-integral-style controller
with Euler integration. It runs for a fixed maximum number of steps and may
stop early when a loss tolerance is reached. It does not guarantee convergence
or a globally optimal control signal.

### Local plasticity phase

After control, the implementation updates weights using controlled and baseline
activities:

$$
\Delta W_i = \eta_w \frac{1}{B}\sum_{b=1}^{B}
\left(a_{i,b}^{\mathrm{ctrl}} - a_{i,b}^{\mathrm{base}}\right)
\left(a_{i-1,b}^{\mathrm{pre}}\right)^T.
$$

Biases are updated using the minibatch mean of the same postsynaptic activity
difference. For deeper layers, the previous layer's controlled activity is
used as the presynaptic activity.

This is close in spirit to the delta-style Target Learning rule. It is not
identical to the canonical DFC update, which compares controlled postsynaptic
activity with the feedforward compartment activity.

## Dendritic Effects

The model supports two modes through `dendritic_effect`.

### Additive mode

```text
target_activation = leaky_relu(z + tanh(c))
```

This is the closer analogue of the additive preactivation feedback used in the
DFC formulation.

### Multiplicative mode

```text
target_activation = (1 + tanh(c)) * relu(z)
```

This makes the top-down signal a gain on bottom-up activity. It is a
project-specific hypothesis, not a mechanism established by the included
literature. The hard ReLU derivative is zero for inactive units, which can
affect controllability and learning stability.

## Repository Structure

```text
src/
  train.py                 Main single-configuration training entry point
  evaluation.py            Matched additive/multiplicative MNIST evaluation
  core/
    controllers.py         Control optimization and diagnostics
    euler_integrators.py   Neural and controller Euler dynamics
    plasticity.py          Local weight and bias updates
    trainer.py              Training-loop orchestration
  data/
    mnist/dataset.py       MNIST loading and continuous targets
    xor/dataset.py         XOR data loading
  models/network.py        Network and population dynamics
  utils/
    config.py              Experiment dataclasses and defaults
    utils.py               Reproducibility, logging, and checkpoints
  visualizations/           Visualization helpers
tests/                      Unit and integration tests
literature/                 Included reference papers
```

## Requirements and Installation

- Python 3.10 or newer
- PyTorch and torchvision
- NumPy, Matplotlib, and pytest

Install the pinned environment from [requirements.txt](requirements.txt):

```bash
python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
python -m pip install -r requirements.txt
```

The MNIST loader downloads the dataset automatically when it is not present.
By default, it stores the data under `data/MNIST`.

### Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash
python -m pip install -r requirements.txt
python src/train.py
```

The first MNIST run may take several minutes on a CPU. The initial dataset
download requires internet access; after that, dataset files and training logs
are written locally and no experiment-tracking service is required.

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover controller behavior, Euler integration, local plasticity,
state management, training updates, and MNIST batch formatting. They do not yet
constitute a full multi-seed MNIST benchmark.

If imports fail, confirm that the virtual environment is active and that the
command is being run from the repository root. In Windows PowerShell, activate
the environment with `.venv\Scripts\Activate.ps1` instead of the Git Bash
command above.

## Running a Single Experiment

From the repository root, with the virtual environment active:

```bash
python src/train.py
```

The current entry point uses:

- Architecture: `[784, 256, 128, 64, 10]`
- Controller: `pid` configuration, five maximum control steps
- Plasticity learning rate: `1e-4`
- Batch size: `64`
- Epochs: `10`
- Seed: `7`
- Dendritic effect: `multiplicative`

These values are defined in [src/train.py](src/train.py) and
[src/utils/config.py](src/utils/config.py). Modify those files to change the
single-run experiment.

Running `src/train.py` prints final test MSE and classification accuracy after
training. The default run does not save a checkpoint.

## Reproducible Evaluation

Use [src/evaluation.py](src/evaluation.py) to compare additive and
multiplicative effects with matched settings:

```bash
python src/evaluation.py --epochs 10 --seeds 7 17 27 37 47
```

For a faster smoke test:

```bash
python src/evaluation.py --epochs 1 --seeds 7
```

The evaluator keeps the test split out of optimization and records per epoch:

- Training MSE
- Held-out test MSE
- Held-out classification accuracy
- Mean control magnitude and improvement
- Control failure rate
- Maximum forward-weight norm
- Finite-value status

Results are written to `evaluation_results/`:

```text
mnist_results.json          Complete configurations and histories
mnist_epoch_metrics.csv     Tabular epoch-level metrics
mnist_learning_curves.png   Mean curves with across-seed variation
mnist_final_accuracy.png    Per-seed final accuracy comparison
```

Classification accuracy is the primary performance metric. MSE is a secondary
diagnostic because targets are continuous one-hot-like vectors with target
values `1.0` and `0.05`.


## Known Limitations

- The implementation uses direct dataset targets during control rather than
  the nudged output target used in the canonical DFC derivation.
- Feedback matrices $Q_i$ are initialized from forward weights and then frozen;
  learned-feedback DFC is not implemented.
- The controller is PI-like despite the configuration name `pid`;
  `use_derivative` applies activation-derivative modulation to feedback rather
  than adding a conventional PID derivative term.
- The neuron dynamics and dendritic modulation differ from canonical DFC
  preactivation dynamics.
- Long-run numerical stability is not guaranteed.
- Prospective Configuration is conceptual background only; it is not a
  separately implemented algorithm in the current source tree.
- Checkpointing and optional Weights & Biases logging are disabled by default.


## Literature

The repository includes these reference documents in `literature/`:

- *Challenging Backpropagation: Evidence for Target Learning in the Neocortex*,
  Pau Vilimelis Aceituno et al., bioRxiv preprint.
- *Credit Assignment in Neural Networks through Deep Feedback Control*,
  Alexander Meulemans et al., NeurIPS 2021.

This project should be cited as an implementation inspired by these works, not as an exact reproduction of their models or experimental results.

## How to Cite This Repository

tbd