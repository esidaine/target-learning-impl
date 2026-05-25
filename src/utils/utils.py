import torch
import numpy as np
import random
import logging
import os
import inspect
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
from core.euler_integrators import ControlErrorIntegrator
import torch.nn.functional as F

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

def log_feedback_alignment(network, global_control: torch.Tensor) -> None:
    dfc_controls = network.DFC_project_feedback(global_control, use_derivative=False)
    bp_controls  = network.chain_rule_project_feedback(global_control)

    for i, (c_dfc, c_bp) in enumerate(zip(dfc_controls, bp_controls)):
        cos_sim = F.cosine_similarity(
            c_dfc.flatten(1), c_bp.flatten(1), dim=1
        ).mean().item()
        network.stats[f'feedback_alignment_layer_{i}'] = cos_sim


def save_experiment(network, controller, plasticity, epoch, loss, task):
    """
    Saves the model weights and experiment metadata.
    """
    # 1. Create a dedicated directory if it doesn't exist
    save_dir = Path("weights")
    save_dir.mkdir(parents=True, exist_ok=True) 
    

    # Forcefully remove hidden spaces, newlines, and carriage returns
    clean_task = str(task).strip().replace("\n", "").replace("\r", "")
    clean_mode = str(controller.mode).strip().replace("\n", "").replace("\r", "")

    json_filename = f"{clean_task}_{clean_mode}_best_model_meta.json"
    json_path = save_dir / json_filename
    
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

