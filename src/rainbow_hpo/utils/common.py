"""
Common utility functions used across the Rainbow HPO project.
"""

import os
import random
import numpy as np
import torch
import logging
import sys
import warnings
import optuna
import tqdm
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

# Global set to track repeated warnings
_seen_warnings = set()

class DuplicateFilter(logging.Filter):
    """Filter to prevent duplicate log messages."""
    
    def __init__(self):
        super().__init__()
        self.seen_msgs = set()
        
    def filter(self, record):
        msg = record.getMessage()
        if msg in self.seen_msgs:
            return False  # Reject duplicate message
        self.seen_msgs.add(msg)
        return True

class ContextAdapter(logging.LoggerAdapter):
    """Add context information to log records."""
    
    def process(self, msg, kwargs):
        ctx_str = ' '.join(f'{k}={v}' for k, v in self.extra.items())
        if ctx_str:
            return f"[{ctx_str}] {msg}", kwargs
        return msg, kwargs

class TqdmLoggingHandler(logging.Handler):
    """Handler that writes logs compatible with tqdm progress bars."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def suppress_repeated_warnings(message: str):
    """
    Suppress warnings that have been seen before.
    
    Args:
        message: Warning message
    """
    global _seen_warnings
    if message in _seen_warnings:
        return True  # Already seen this warning
    _seen_warnings.add(message)
    return False

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


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO, run_id: str = None, trial_id: Optional[str] = None):
    """
    Configure logging for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (e.g., logging.INFO)
        run_id: Optional identifier for this run to include in logs
        trial_id: Optional trial identifier for HPO scenarios
    
    Returns:
        run_id: The run identifier (generated if not provided)
    """
    # Reset global state to avoid issues with duplicate filters
    global _seen_warnings
    _seen_warnings = set()
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate run_id if not provided
    if not run_id:
        import datetime
        import uuid
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        run_id = f"{timestamp}_{short_uuid}"
    
    # Create context string for log format
    context_parts = [f"run={run_id}"]
    if trial_id is not None:
        context_parts.append(f"trial={trial_id}")
    
    context_str = " ".join(context_parts)
    log_format = f"%(asctime)s [%(levelname)s] [{context_str}] %(name)s: %(message)s"
    
    # Configure root logger - use TqdmLoggingHandler for console to be compatible with progress bars
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates on multiple calls
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Remove any existing filters to avoid issues with stateful filters
    for filter in list(root_logger.filters):
        root_logger.removeFilter(filter)
    
    # Configure console handler with tqdm-compatible output
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Configure file handler for all logs
    log_file_name = f"main_{run_id}"
    if trial_id is not None:
        log_file_name += f"_trial{trial_id}"
    log_file_name += ".log"
    
    main_file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    main_file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(main_file_handler)
    
    # Add duplicate filter to root logger
    duplicate_filter = DuplicateFilter()
    root_logger.addFilter(duplicate_filter)
    
    # Configure specific component loggers - store in separate files but don't duplicate console output
    components = {
        "agent_builder": "Agent building",
        "env_builder": "Environment building",
        "training": "Training process",
        "hpo": "Hyperparameter optimization"
    }
    
    for name, description in components.items():
        component_logger = logging.getLogger(name)
        component_logger.propagate = True  # Let root logger handle console output
        component_logger.setLevel(log_level)
        
        # Remove any existing handlers
        for handler in list(component_logger.handlers):
            component_logger.removeHandler(handler)
        
        # Component-specific file handler
        comp_file_name = f"{name}_{run_id}"
        if trial_id is not None:
            comp_file_name += f"_trial{trial_id}"
        comp_file_name += ".log"
        
        file_handler = logging.FileHandler(os.path.join(log_dir, comp_file_name))
        file_handler.setFormatter(logging.Formatter(log_format))
        component_logger.addHandler(file_handler)
    
    # Configure third-party libraries
    
    # Optuna
    try:
        # Set Optuna's own verbosity
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        # Redirect Optuna logging to our system
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.propagate = True  # Let root logger handle console output
        optuna_logger.setLevel(log_level)
        
        # Remove any existing handlers
        for handler in list(optuna_logger.handlers):
            optuna_logger.removeHandler(handler)
        
        # Add separate file handler for optuna
        optuna_file_name = f"optuna_{run_id}"
        if trial_id is not None:
            optuna_file_name += f"_trial{trial_id}"
        optuna_file_name += ".log"
        
        optuna_file_handler = logging.FileHandler(os.path.join(log_dir, optuna_file_name))
        optuna_file_handler.setFormatter(logging.Formatter(log_format))
        optuna_logger.addHandler(optuna_file_handler)
        
        # Use our logger for Optuna
        optuna_handler = logging.getLogger("optuna")
    except Exception as e:
        logger.warning(f"Failed to configure Optuna logging: {e}")
    
    # Silence chatty third-party libraries by default
    for lib_name in ["gymnasium", "torch", "matplotlib", "PIL"]:
        try:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.setLevel(logging.WARNING)
        except Exception:
            pass
    
    # Log startup information
    logger.info(f"Logging configured with run_id={run_id}" + 
                (f", trial_id={trial_id}" if trial_id is not None else "") +
                f". Log files stored in: {log_dir}")
    
    return run_id

def get_context_logger(name: str, **context):
    """
    Get a logger with additional context information.
    
    Args:
        name: Logger name
        context: Additional context data as keyword arguments (e.g., trial_id=5)
    
    Returns:
        A logger adapter that includes the context information
    """
    return ContextAdapter(logging.getLogger(name), context)

# Monkey patch the warnings.showwarning function to suppress repeated warnings
original_showwarning = warnings.showwarning

def showwarning_with_suppression(message, category, filename, lineno, file=None, line=None):
    """Replacement for warnings.showwarning that suppresses repeated messages."""
    msg_str = str(message)
    if not suppress_repeated_warnings(msg_str):
        original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = showwarning_with_suppression