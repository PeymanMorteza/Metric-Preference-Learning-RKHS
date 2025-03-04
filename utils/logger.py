import logging
import os
from datetime import datetime

def setup_logger(experiment_name="experiment",log_dir="logs"):
    """
    Sets up a logger that writes logs to a file and displays them in the console.

    Args:
        experiment_name (str): Name of the experiment to include in the log filename.
    """
    # Create logs directory if it doesn't exist
    #log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique log file with timestamp
    log_filename = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        filename=log_filename, 
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"  # Overwrites old logs; use "a" to append
    )

    # Console handler (so logs appear in the terminal as well)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add console handler to the logger
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logger initialized for experiment: {experiment_name}")
