"""
Common utility functions used across the Rainbow HPO project.
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def create_directory_if_not_exists(path: str) -> None:
    """
    Create directory if it does not already exist.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (e.g., logging.INFO)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "main.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    loggers = {
        "agent_builder": os.path.join(log_dir, "agent_builder.log"),
        "env_builder": os.path.join(log_dir, "env_builder.log"),
        "training": os.path.join(log_dir, "training.log"),
        "hpo": os.path.join(log_dir, "hpo.log")
    }
    
    for name, log_file in loggers.items():
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        
    logger.info(f"Logging configured. Log files stored in: {log_dir}")