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
        # Create the format (Notice I added [%(name)s] so it prints the file name!)
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