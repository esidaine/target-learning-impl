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
    
    # If called from a script, it gets the filename. 
    # If called from a notebook, it usually defaults to '__main__'
    logger_name = caller_module.__name__ if caller_module else "root"

    # 2. Configure the logger
    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"), # Saves to a file
            logging.StreamHandler()              # Prints to terminal
        ]
    )
    return logging.getLogger()