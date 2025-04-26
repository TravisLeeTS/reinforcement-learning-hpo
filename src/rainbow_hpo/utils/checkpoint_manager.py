"""
Checkpoint manager for hyperparameter optimization.
Provides functionality to save and restore optimization state.
"""

import os
import json
import pickle
import time
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing and restoration of HPO experiment state.
    Supports multiple serialization formats and recovery strategies.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 experiment_name: str = None,
                 interval: int = 10,
                 keep_history: int = 3,
                 verbose: bool = True):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            experiment_name: Name for this experiment (defaults to timestamp)
            interval: How many trials between checkpoints
            keep_history: Number of checkpoint versions to keep
            verbose: Whether to log checkpoint operations
        """
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name or f"hpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.interval = interval
        self.keep_history = keep_history
        self.verbose = verbose
        
        # Create checkpoint directory
        self.experiment_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        if self.verbose:
            logger.info(f"Checkpoint manager initialized at {self.experiment_dir}")
    
    def save(self, state: Dict[str, Any], trial_idx: int) -> str:
        """
        Save optimization state to checkpoint.
        
        Args:
            state: Dictionary containing optimizer state
            trial_idx: Current trial index
            
        Returns:
            Path to the saved checkpoint
        """
        # Only checkpoint at specified intervals
        if trial_idx % self.interval != 0 and trial_idx != -1:
            return ""
        
        # Prepare checkpoint name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{trial_idx}_{timestamp}"
        
        # Create full paths for different formats
        pkl_path = os.path.join(self.experiment_dir, f"{checkpoint_name}.pkl")
        json_path = os.path.join(self.experiment_dir, f"{checkpoint_name}.json")
        
        # Make a copy of state to avoid modifying the original
        state_copy = state.copy()
        
        # Ensure timestamp is included
        if "timestamp" not in state_copy:
            state_copy["timestamp"] = time.time()
        
        if "checkpoint_metadata" not in state_copy:
            state_copy["checkpoint_metadata"] = {}
        
        state_copy["checkpoint_metadata"].update({
            "trial_idx": trial_idx,
            "timestamp": timestamp,
            "format_version": "1.0"
        })
        
        # Save in pickle format for full state (primary)
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(state_copy, f)
            
            if self.verbose:
                logger.info(f"Saved checkpoint to {pkl_path}")
        except Exception as e:
            logger.error(f"Error saving pickle checkpoint: {e}")
        
        # Also save a human-readable JSON with essential information
        try:
            # Extract json-serializable information
            json_state = {
                "best_params": state_copy.get("best_params", {}),
                "best_value": float(state_copy.get("best_value", float("-inf"))),
                "n_trials_completed": state_copy.get("n_completed", trial_idx),
                "timestamp": timestamp,
                "checkpoint_path": pkl_path
            }
            
            # Add any custom metrics if present
            if "metrics" in state_copy:
                json_state["metrics"] = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in state_copy["metrics"].items()
                }
            
            with open(json_path, "w") as f:
                json.dump(json_state, f, indent=2)
            
            # Create a symbolic link to the latest checkpoint
            latest_json = os.path.join(self.experiment_dir, "latest_checkpoint.json")
            latest_pkl = os.path.join(self.experiment_dir, "latest_checkpoint.pkl")
            
            # Remove existing links if they exist
            if os.path.exists(latest_json):
                os.remove(latest_json)
            if os.path.exists(latest_pkl):
                os.remove(latest_pkl)
            
            # Copy the files to the latest links
            with open(json_path, "r") as src, open(latest_json, "w") as dst:
                dst.write(src.read())
            
            import shutil
            shutil.copy2(pkl_path, latest_pkl)
            
            if self.verbose:
                logger.info(f"Saved JSON summary to {json_path}")
        
        except Exception as e:
            logger.error(f"Error saving JSON checkpoint summary: {e}")
        
        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        return pkl_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the history limit."""
        if self.keep_history <= 0:
            return
        
        # List all pickle checkpoints
        checkpoints = [f for f in os.listdir(self.experiment_dir) 
                      if f.startswith("checkpoint_") and f.endswith(".pkl")]
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda f: os.path.getmtime(
            os.path.join(self.experiment_dir, f)), reverse=True)
        
        # Remove old checkpoints
        for old_ckpt in checkpoints[self.keep_history:]:
            try:
                pkl_path = os.path.join(self.experiment_dir, old_ckpt)
                json_path = pkl_path.replace(".pkl", ".json")
                
                # Remove both formats if they exist
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                if os.path.exists(json_path):
                    os.remove(json_path)
                
                if self.verbose:
                    logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.error(f"Error removing old checkpoint {old_ckpt}: {e}")
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Dictionary with optimizer state or None if no checkpoint found
        """
        return self.load()
    
    def load(self, checkpoint_path: str = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from specified path or latest if not specified.
        
        Args:
            checkpoint_path: Path to specific checkpoint or None for latest
            
        Returns:
            Dictionary with optimizer state or None if no checkpoint found
        """
        if not checkpoint_path:
            # Try to load the latest checkpoint
            checkpoint_path = os.path.join(self.experiment_dir, "latest_checkpoint.pkl")
            
            if not os.path.exists(checkpoint_path):
                # Look for most recent checkpoint if latest link doesn't exist
                checkpoints = [f for f in os.listdir(self.experiment_dir) 
                              if f.startswith("checkpoint_") and f.endswith(".pkl")]
                
                if not checkpoints:
                    if self.verbose:
                        logger.warning(f"No checkpoints found in {self.experiment_dir}")
                    return None
                
                # Sort by modification time (newest first)
                checkpoints.sort(key=lambda f: os.path.getmtime(
                    os.path.join(self.experiment_dir, f)), reverse=True)
                
                checkpoint_path = os.path.join(self.experiment_dir, checkpoints[0])
        
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            
            if self.verbose:
                trial_idx = checkpoint.get("checkpoint_metadata", {}).get("trial_idx", "?")
                logger.info(f"Loaded checkpoint from {checkpoint_path} (trial {trial_idx})")
            
            return checkpoint
        
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            
            # Try to recover from JSON if pickle fails
            try:
                json_path = checkpoint_path.replace(".pkl", ".json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        json_checkpoint = json.load(f)
                    
                    logger.warning("Recovered partial state from JSON backup")
                    return json_checkpoint
            except Exception as json_error:
                logger.error(f"Also failed to load JSON backup: {json_error}")
            
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        # Look for all checkpoints
        for filename in os.listdir(self.experiment_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                try:
                    json_path = os.path.join(self.experiment_dir, filename)
                    with open(json_path, "r") as f:
                        info = json.load(f)
                    
                    # Add the filename
                    info["filename"] = filename
                    info["path"] = json_path.replace(".json", ".pkl")
                    
                    checkpoints.append(info)
                except Exception as e:
                    logger.error(f"Error reading checkpoint info {filename}: {e}")
        
        # Sort by trial number
        checkpoints.sort(key=lambda x: x.get("n_trials_completed", 0))
        
        return checkpoints
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment.
        
        Returns:
            Dictionary with experiment information
        """
        # Get all checkpoints
        checkpoints = self.list_checkpoints()
        
        # Find the best result across all checkpoints
        best_value = float("-inf")
        best_params = {}
        
        for ckpt in checkpoints:
            value = ckpt.get("best_value", float("-inf"))
            if value > best_value:
                best_value = value
                best_params = ckpt.get("best_params", {})
        
        # Count total trials
        n_trials = max([ckpt.get("n_trials_completed", 0) for ckpt in checkpoints]) if checkpoints else 0
        
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "n_checkpoints": len(checkpoints),
            "n_trials": n_trials,
            "best_value": best_value,
            "best_params": best_params,
            "checkpoint_interval": self.interval,
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        }


# Standalone test code
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Test the checkpoint manager
    manager = CheckpointManager(checkpoint_dir="test_checkpoints",
                              experiment_name="test_experiment")
    
    # Create a test state
    state = {
        "best_params": {"learning_rate": 0.01, "batch_size": 64},
        "best_value": 0.85,
        "n_completed": 10,
        "results": [
            {"trial": i, "value": i / 10} for i in range(10)
        ]
    }
    
    # Save test state
    manager.save(state, 10)
    
    # Load it back
    loaded_state = manager.load_latest()
    
    if loaded_state:
        print("Successfully loaded checkpoint!")
        print(f"Best value: {loaded_state.get('best_value')}")
    
    # List all checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Get experiment info
    info = manager.get_experiment_info()
    print(f"Experiment: {info['experiment_name']}")
    print(f"Total trials: {info['n_trials']}")