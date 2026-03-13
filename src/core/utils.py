import torch
import numpy as np
import random
import logging
import inspect

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
            error = abs(result - expected_outputs)
            if error <= tolerance:
                print(f"  [PASS] Test {i+1}: Output {result:.5f} matched expected {expected_outputs}")
                passed += 1
            else:
                print(f"  [FAIL] Test {i+1}: Expected {expected_outputs}, got {result:.5f} (Error: {error})")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] Test {i+1} crashed with inputs {inputs}. Error: {e}")
            failed += 1
            
    print(f"--- Results: {passed} Passed | {failed} Failed ---\n")
    return failed == 0