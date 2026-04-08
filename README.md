# Target Learning Implementation

[![Status](https://img.shields.io/badge/Status-Ongoing_Research-blue.svg)]()
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-Supported-yellow.svg)]()

## 📌 Overview

This repository implements a biologically plausible framework for cortical learning based on **Target Learning**  *(Vilimelis Aceituno et al, 2024)*, **Deep Feedback Control (DFC)** *(Meulemans et al., 2021)*, and **Prospective Configuration** *(Song et al., 2024)*. 

Here, a top-down feedback controller (simulated here via PID dynamics or gradient-based search) modulates the dendritic compartments of hidden neurons to drive their somatic firing rates toward a desired equilibrium. Synaptic plasticity then acts locally in time and space, pulling the baseline feedforward activity toward dynamically inferred targets.

## Core Architecture & Code Structure

### 1. Multi-Compartment Network (`Network` & `NeuralPopulation`)
* Models pyramidal neurons receiving distinct bottom-up sensory input and top-down control signals .
* The integrated firing rate combines these components multiplicatively.

### 2. The Control Mechanism (`ControlMechanism`)
* Generates control signals per neuron to find the optimal activation state.
* Supports two modes:
  * `pid`: Models a biological dynamical system where the controller continuously nudges neurons using leaky integrators until they settle into an optimal equilibrium.
  * `backprop`: An iterative, gradient-based baseline approach to finding the optimal control signals.

### 3. Local Plasticity (`Plasticity`)
* Updates network weights utilizing a fully local, Hebbian-like learning rule. 
* The difference between the baseline (control-free) activation ($a_{baseline}$) and the controlled target state ($a_{controlled}$) dictates the parameter update: $\Delta W = \frac{1}{B} \sum (a_{controlled} - a_{baseline})^T \cdot a_{pre}$.

### 4. Training Loop (`Trainer`)
* Orchestrates epochs through three discrete phases:
  1. **Baseline Phase**: Forward pass with zero top-down control ($c_n = 0$) to measure the natural network response.
  2. **Control Phase**: Optimization of the control signals via dynamic inversion until the network reaches the target physiological state.
  3. **Plasticity Phase**: Local weight adjustments based on the discrepancies computed in the control phase.
