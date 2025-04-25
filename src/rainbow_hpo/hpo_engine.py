"""
Hyperparameter optimization engine for Rainbow DQN project.
Implements various search strategies for finding optimal hyperparameters.
"""

import numpy as np
import logging
import random
from typing import Dict, Any, Callable, Tuple, List, Optional, Union
import copy
import time
from dataclasses import dataclass
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/hpo_engine.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a hyperparameter optimization trial."""
    params: Dict[str, Any]
    value: float
    trial_id: int
    duration: float


class HyperparameterOptimizer:
    """
    Hyperparameter optimization engine for Rainbow DQN.
    Implements a simple but effective TPE (Tree-structured Parzen Estimator) inspired approach.
    """
    
    def __init__(
        self,
        param_space: Dict[str, Any],
        objective_function: Callable[[Dict[str, Any], int], float],
        n_trials: int = 10,
        seed: int = 42,
        strategy: str = "tpe"
    ):
        """
        Initialize the HyperparameterOptimizer.
        
        Args:
            param_space: Dictionary defining hyperparameter space
                Each entry should be a tuple (distribution_type, *args)
                Supported distributions:
                - "uniform": (min_value, max_value)
                - "log_uniform": (min_value, max_value)
                - "int": (min_value, max_value)
                - "categorical": [option1, option2, ...]
            objective_function: Function to maximize, takes params dict and trial_id
            n_trials: Number of trials to run
            seed: Random seed
            strategy: Optimization strategy ("random", "tpe")
        """
        self.param_space = param_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.strategy = strategy
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Trial history
        self.trials: List[TrialResult] = []
        
        logger.info(f"Initialized HyperparameterOptimizer with strategy '{strategy}'")
        logger.info(f"Parameter space: {param_space}")
    
    def _sample_params_random(self) -> Dict[str, Any]:
        """
        Sample parameters randomly from the parameter space.
        
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_config in self.param_space.items():
            dist_type = param_config[0]
            
            if dist_type == "uniform":
                min_val, max_val = param_config[1], param_config[2]
                params[param_name] = random.uniform(min_val, max_val)
            
            elif dist_type == "log_uniform":
                min_val, max_val = param_config[1], param_config[2]
                log_min, log_max = np.log(min_val), np.log(max_val)
                params[param_name] = float(np.exp(random.uniform(log_min, log_max)))
            
            elif dist_type == "int":
                min_val, max_val = param_config[1], param_config[2]
                params[param_name] = random.randint(min_val, max_val)
            
            elif dist_type == "categorical":
                options = param_config[1]
                params[param_name] = random.choice(options)
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        return params
    
    def _sample_params_tpe(self, best_frac: float = 0.2) -> Dict[str, Any]:
        """
        Sample parameters using a simplified TPE approach.
        
        Args:
            best_frac: Fraction of trials to consider as "good"
            
        Returns:
            Dictionary of sampled parameters
        """
        # For the first few trials, use random sampling
        if len(self.trials) < 5:
            return self._sample_params_random()
        
        # Split trials into good and bad based on performance
        sorted_trials = sorted(self.trials, key=lambda t: t.value, reverse=True)
        n_good = max(1, int(len(sorted_trials) * best_frac))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]
        
        # If not enough trials in either group, fall back to random sampling
        if len(good_trials) == 0 or len(bad_trials) == 0:
            return self._sample_params_random()
        
        params = {}
        
        for param_name, param_config in self.param_space.items():
            dist_type = param_config[0]
            
            # Extract values for this parameter from trials
            good_values = [t.params[param_name] for t in good_trials]
            bad_values = [t.params[param_name] for t in bad_trials]
            
            if dist_type == "uniform" or dist_type == "log_uniform":
                # For continuous parameters, fit simple Gaussian KDE
                # We'll use the mean and std of good trials to sample new values
                mean = np.mean(good_values)
                std = np.std(good_values) if len(good_values) > 1 else (param_config[2] - param_config[1]) / 10
                
                # Add some noise to avoid getting stuck
                std = max(std, (param_config[2] - param_config[1]) / 20)
                
                # Sample new value
                value = np.random.normal(mean, std)
                
                # Clip to parameter range
                value = max(param_config[1], min(param_config[2], value))
                
                # For log_uniform, convert back to original scale if we sampled in log space
                params[param_name] = float(value)
            
            elif dist_type == "int":
                # For integer parameters, sample from categorical distribution based on frequencies
                # Here we're simplifying by just sampling around the mean
                mean = np.mean(good_values)
                std = max(1, np.std(good_values) if len(good_values) > 1 else (param_config[2] - param_config[1]) / 4)
                
                value = int(round(np.random.normal(mean, std)))
                value = max(param_config[1], min(param_config[2], value))
                params[param_name] = value
            
            elif dist_type == "categorical":
                # For categorical parameters, sample based on frequency in good trials
                options = param_config[1]
                
                # Count occurrences of each option in good trials
                counts = {option: good_values.count(option) + 0.5 for option in options}  # Add small constant to avoid zeros
                
                # Normalize to get probabilities
                total = sum(counts.values())
                probs = [counts[option] / total for option in options]
                
                # Sample new value
                params[param_name] = np.random.choice(options, p=probs)
        
        return params
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Run the hyperparameter optimization process.
        
        Returns:
            Tuple of (best_params, best_value)
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        for trial_id in range(self.n_trials):
            logger.info(f"Starting trial {trial_id + 1}/{self.n_trials}")
            
            # Sample parameters
            if self.strategy == "random":
                params = self._sample_params_random()
            elif self.strategy == "tpe":
                params = self._sample_params_tpe()
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Run objective function
            start_time = time.time()
            try:
                value = self.objective_function(params, trial_id)
            except Exception as e:
                logger.error(f"Trial {trial_id} failed with error: {e}")
                value = float('-inf')  # Failed trials are considered worst
            
            duration = time.time() - start_time
            
            # Record result
            result = TrialResult(
                params=copy.deepcopy(params),
                value=value,
                trial_id=trial_id,
                duration=duration
            )
            self.trials.append(result)
            
            logger.info(f"Trial {trial_id} completed with value {value:.4f} in {duration:.2f}s")
            logger.info(f"Parameters: {params}")
        
        # Get best result
        best_trial = max(self.trials, key=lambda t: t.value)
        
        logger.info(f"Optimization completed. Best value: {best_trial.value:.4f}")
        logger.info(f"Best parameters: {best_trial.params}")
        
        return best_trial.params, best_trial.value
    
    def get_trial_history(self) -> List[TrialResult]:
        """
        Get the history of all trials.
        
        Returns:
            List of TrialResult objects
        """
        return self.trials


def example_objective(params: Dict[str, Any], trial_id: int) -> float:
    """
    Example objective function for demonstration.
    
    Args:
        params: Hyperparameters
        trial_id: Trial ID
        
    Returns:
        Objective value (to be maximized)
    """
    # Simple quadratic function with some noise
    x = params["x"]
    y = params["y"]
    
    # Objective is maximum at x=0, y=5
    value = -(x**2) - (y - 5)**2 + random.uniform(-0.1, 0.1)
    
    return value


if __name__ == "__main__":
    # Simple example to verify the optimizer works
    param_space = {
        "x": ("uniform", -10, 10),
        "y": ("uniform", 0, 10),
    }
    
    # Initialize optimizer
    hpo = HyperparameterOptimizer(
        param_space=param_space,
        objective_function=example_objective,
        n_trials=20,
        strategy="tpe"
    )
    
    # Run optimization
    best_params, best_value = hpo.optimize()
    
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")