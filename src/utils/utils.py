import torch
import numpy as np
import random
import logging
import os
import inspect
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
from core.euler_integrators import ControlErrorIntegrator

def get_weight_metrics(network): 
    """
    Extracts weight norms and histograms for W&B logging.
    Returns a dictionary of metrics.
    """
    metrics = {}
    for i, pop in enumerate(network.populations):
        weights = pop.W.weight.detach().cpu()
        
        metrics[f"Weight_Norms/Pop_{i}"] = weights.norm().item()
        metrics[f"Weight_Histograms/Pop_{i}"] = wandb.Histogram(weights.numpy())
        
    return metrics


def save_experiment(network, controller, plasticity, epoch, loss, task):
    """
    Saves the model weights and experiment metadata.
    """
    # 1. Create a dedicated directory if it doesn't exist
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    json_filename = f"{task}_{controller.mode}_best_model_meta.json"
    json_path = os.path.join(save_dir, json_filename)
    
    # 3. Harvest the Hyperparameters (The "Context")
    hyperparameters = {
        "architecture": {
            "pop_sizes":[network.populations[0].W.in_features] + [pop.num_neurons for pop in network.populations],
        },
        "mechanics": {
            "mode": controller.mode,
            "lr_c": controller.lr_c,
            "max_steps": controller.max_steps,
            "lr_theta": plasticity.lr_theta
        },
        "training": {
            "epoch_reached": epoch,
            "final_loss": float(loss),
            "seed": 7 # Hardcoded for reproducibility, but could be made dynamic if needed
        }
    }

    # 4. Harvest the Weights
    # Assuming populations have a 'synapses' attribute with weights/biases
    model_state = {
        'weights': [pop.W.weight.data.clone() for pop in network.populations],
    }

    # 5. Save the pure JSON metadata (For human readability and quick scanning)
    with open(json_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # 6. Save the PyTorch Checkpoint (Contains everything needed to resume)
    checkpoint = {
        "hyperparameters": hyperparameters,
        "model_state": model_state
    }
    pth_filename = f"{task}_{controller.mode}_best_model_meta.pth"
    pth_path = os.path.join(save_dir, pth_filename)
    torch.save(checkpoint, pth_path)
    # Tell W&B to upload these specific files to the cloud! ---
    if wandb.run is not None:  # Check if W&B is active
        wandb.save(json_path, base_path=save_dir)
        wandb.save(pth_path, base_path=save_dir)


def set_all_seeds(seed=42):
    """Locks all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seeds locked to {seed}")

def get_logger():
    """
    Automatically detects the filename of the caller and sets up logging.
    """
    # 1. Grab the name of the file that called this function
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    logger_name = caller_module.__name__ if caller_module else "root"

    # 2. Get the specific logger for this file
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 3. Prevent duplicate logs if called multiple times (crucial for notebooks)
    if not logger.handlers:
        # Create the format
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
        
        # File Handler (Saves to training.log)
        file_handler = logging.FileHandler("training.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream Handler (Prints to terminal/notebook)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 4. Return the specific logger we just built
    return logger


def test_function(func, test_cases, tolerance=1e-5):
    """
    Expects test cases to be a list of tuples which carry a dictionary of inputs and the expected output.
    e.g. 
    test_cases = [
        # Tuple 1 (Test Case 1)
        ( {"z_n": 0.5, "c_n": 0.2, "beta": 1.0},  1.0 ),
        
        # Tuple 2 (Test Case 2)
        ( {"z_n": 0.0, "c_n": 100.0, "beta": 2.0}, 0.0 )
    ]
     """
    
    print(f"--- Running tests for: {func.__name__} ---")
    
    passed = 0
    failed = 0

    for i, (inputs, expected_outputs) in enumerate(test_cases): # grab a test case (touple) and unpack inputs (dict) and expected outputs
        try:
            # Call the function with the inputs not as dict, but unpacked as keyword arguments
            result = func(**inputs) 
            # Check if the absolute difference is within our tolerance (ignores sign)
            error_array = np.abs(result - expected_outputs)
            
            if np.allclose(result, expected_outputs, atol=tolerance):
                print(f"  [PASS] Test {i+1}: Output {result:.5f} matched expected {expected_outputs}")
                passed += 1
            else:
                print(f"  [FAIL] Test {i+1}: Expected {expected_outputs}, got {result:.5f} (Error: {error_array})")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] Test {i+1} crashed with inputs {inputs}. Error: {e}")
            failed += 1
            
    print(f"--- Results: {passed} Passed | {failed} Failed ---\n")
    return failed == 0

def debug_dynamical_settling(network, controller, sensory_inputs, target_y):
    """
    Run this function on a single batch (or single data point) to visualize 
    if your dt, tau, and k_p are mathematically stable.
    """
    network.eval()
    
    # Tracking lists
    errors = []
    control_magnitudes = []
    neuron_activations = []

    with torch.no_grad():
        baseline_pred = network(sensory_inputs, control_signals=None, save_baseline=True)

    # Initialize Controller variables
    batch_size = sensory_inputs.size(0)
    output_size = target_y.size(1)
    global_control = torch.zeros(batch_size, output_size)
    global_control_integral = torch.zeros(batch_size, output_size)
    control_stepper = ControlErrorIntegrator(dt=0.1, tau=1.0, alpha=0.1, k_p=0.05)
    
    y_pred = baseline_pred

    for step in range(100): # Force a longer run to see the full curve
        output_error = target_y - y_pred 
        mse_loss = torch.mean(output_error ** 2).item()
        
        # Step the controller
        global_control_integral, global_control = control_stepper.step(
            global_control, global_control_integral, output_error) 
        
        # Step the network
        local_controls = network.get_local_controls(global_control)
        y_pred = network(sensory_inputs, control_signals=local_controls, save_baseline=False, dynamic_step=True)

        # --- LOGGING ---
        errors.append(mse_loss)
        control_magnitudes.append(torch.mean(torch.abs(global_control)).item())
        
        # Track a specific neuron in the first hidden layer to see its physical movement
        tracked_neuron_state = network.populations[0].a_controlled[0, 0].item() # [0, 0] grabs the first neuron in the first population
        neuron_activations.append(tracked_neuron_state)

    # --- PLOTTING ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    axs[0].plot(errors, color='red')
    axs[0].set_title('Global MSE Error (Should smoothly drop to near 0)')
    
    axs[1].plot(control_magnitudes, color='blue')
    axs[1].set_title('Global Control Signal Magnitude (Should increase and not explode)')
    
    axs[2].plot(neuron_activations, color='green')
    axs[2].set_title('Single Neuron Physical State (a_controlled)')
    axs[2].set_xlabel('Integration Steps')
    
    plt.tight_layout()
    plt.show()