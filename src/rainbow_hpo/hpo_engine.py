"""
Hyperparameter Optimization (HPO) engine for Rainbow DQN.
Supports various optimization strategies and parallel execution.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field
import multiprocessing
import threading
import psutil
import signal
import datetime
import random
from pathlib import Path
import copy
import traceback

# Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Ray Tune for distributed optimization
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.suggest.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# GPU support (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Plotting (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import our advanced resource monitoring utility
from rainbow_hpo.utils.resource_monitor import ResourceMonitor

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/hpo.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class EarlyStoppingSignal(Exception):
    """Exception raised to signal early stopping of a trial."""
    pass


@dataclass
class TrialResult:
    """Container for trial results."""
    trial_id: int
    params: Dict[str, Any]
    value: float
    duration: float
    early_stopped: bool = False
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class HPOConfig:
    """Configuration for HPO engine."""
    param_space: Dict[str, Any]
    n_trials: int = 50
    n_parallel: int = 1
    optimization_method: str = "optuna"  # "optuna", "random", "grid", "ray"
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    pruner: str = "median"  # "median", "halving", "none"
    direction: str = "maximize"
    timeout: Optional[int] = None  # seconds
    checkpoint_interval: int = 10  # trials
    checkpoint_dir: str = "checkpoints"
    early_stopping: bool = True
    patience: int = 10  # trials without improvement before early stopping
    min_improvement: float = 0.01  # minimum improvement to reset patience
    monitor_resources: bool = True
    resource_limits: Dict[str, float] = field(default_factory=dict)  # e.g., {"cpu": 0.9, "memory": 0.8, "gpu_memory": 0.8}
    random_seed: Optional[int] = None


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for Rainbow DQN.
    Supports multiple optimization strategies.
    """
    
    def __init__(self, param_space: Dict[str, Any], objective_function: Callable, 
                 n_trials: int = 10, seed: int = 42, strategy: str = "tpe",
                 n_parallel: int = 1, early_stopping: bool = True, 
                 checkpoint_dir: str = "checkpoints", checkpoint_interval: int = 5,
                 monitor_resources: bool = True, resource_log_dir: str = "resource_logs"):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            param_space: Dictionary with parameter spaces
            objective_function: Function to optimize
            n_trials: Number of optimization trials
            seed: Random seed
            strategy: Optimization strategy ('tpe', 'random', 'cmaes', 'grid', 'evolutionary')
            n_parallel: Number of parallel trials to run
            early_stopping: Whether to enable early stopping
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Interval (in trials) for saving checkpoints
            monitor_resources: Whether to monitor system resources
            resource_log_dir: Directory for resource usage logs
        """
        self.param_space = param_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.seed = seed
        self.strategy = strategy
        self.n_parallel = min(n_parallel, multiprocessing.cpu_count())
        self.early_stopping = early_stopping
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.monitor_resources = monitor_resources
        self.resource_log_dir = resource_log_dir
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create resource log directory if monitoring is enabled
        if self.monitor_resources:
            os.makedirs(self.resource_log_dir, exist_ok=True)
        
        # Set up advanced resource monitor with more options
        self.resource_monitor = None
        if monitor_resources:
            self.resource_monitor = ResourceMonitor(
                monitor_gpu=True,  # Enable GPU monitoring
                interval=2.0,      # Monitor every 2 seconds
                log_to_file=True,  # Save logs to file
                log_dir=self.resource_log_dir,
                verbose=False,     # Don't print to console
                warning_threshold=0.85  # Warning at 85% usage
            )
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
        if OPTUNA_AVAILABLE:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.best_params = None
        self.best_value = float('-inf')
        self.results = []
        
        logger.info(f"Initialized HPO with {n_trials} trials using {strategy} strategy")
        logger.info(f"Parallel execution: {self.n_parallel} workers")
        logger.info(f"Early stopping: {early_stopping}")
        
    def _update_best(self, value: float, params: Dict[str, Any]) -> bool:
        """Update best parameters if value is better."""
        if value > self.best_value:
            self.best_value = value
            self.best_params = copy.deepcopy(params)
            return True
        return False
    
    def _save_checkpoint(self, trial_results):
        """Save optimizer state to checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "hpo_checkpoint.pkl")
        
        checkpoint = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "results": trial_results,
            "n_completed": len(trial_results),
            "timestamp": time.time()
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Also save in JSON format for easy inspection
        json_path = os.path.join(self.checkpoint_dir, "hpo_status.json")
        
        json_data = {
            "best_params": self.best_params,
            "best_value": float(self.best_value),
            "n_trials_completed": len(trial_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"Saved checkpoint after {len(trial_results)} trials")
    
    def _load_checkpoint(self):
        """Load optimizer state from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "hpo_checkpoint.pkl")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)
                
                self.best_params = checkpoint["best_params"]
                self.best_value = checkpoint["best_value"]
                completed_trials = checkpoint["n_completed"]
                results = checkpoint["results"]
                
                logger.info(f"Loaded checkpoint with {completed_trials} completed trials")
                logger.info(f"Best value so far: {self.best_value}")
                
                return results
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        
        return []
    
    def _check_resources(self):
        """Check if system resources are still available using our advanced ResourceMonitor."""
        if not self.monitor_resources or not self.resource_monitor:
            return True
        
        try:
            # Get the current usage from our ResourceMonitor
            usage = self.resource_monitor.get_current_usage()
            if not usage:
                return True
            
            cpu_percent = usage['cpu_percent']
            memory_percent = usage['memory_percent']
            
            # Check GPU if available
            gpu_alert = False
            if self.resource_monitor.monitor_gpu:
                gpu_utilization = usage['gpu_utilization']
                gpu_memory_used = usage['gpu_memory_used_mb']
                
                if gpu_utilization > 95:
                    logger.warning(f"Critical GPU utilization: {gpu_utilization:.1f}%")
                    gpu_alert = True
            
            # Log resource usage
            if cpu_percent > 90 or memory_percent > 90 or gpu_alert:
                logger.warning(f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
            
            # Stop if resources are critically low
            if memory_percent > 95:
                logger.error("Critical memory usage! Stopping optimization.")
                return False
                
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
        
        return True
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Tuple of (best_parameters, best_value)
        """
        # Start resource monitoring if enabled
        if self.resource_monitor:
            self.resource_monitor.start()
        
        # Try to load checkpoint
        previous_results = self._load_checkpoint()
        
        # Configure HPO based on selected strategy
        if self.strategy == "evolutionary" and DEAP_AVAILABLE:
            logger.info("Using Evolutionary Algorithm optimization")
            best_params, best_value = self._run_evolutionary_optimization()
        elif self.strategy == "grid":
            logger.info("Using Grid Search optimization")
            best_params, best_value = self._run_grid_search()
        elif self.strategy == "pbt" and RAY_AVAILABLE:
            logger.info("Using Population-Based Training optimization")
            best_params, best_value = self._run_population_based_training()
        else:
            # Default to Bayesian optimization with Optuna
            logger.info(f"Using {'TPE' if self.strategy == 'tpe' else self.strategy.capitalize()} optimization via Optuna")
            best_params, best_value = self._run_optuna_optimization()
        
        # Stop resource monitoring and save plot
        if self.resource_monitor:
            # Generate resource usage plot
            plot_path = os.path.join(self.resource_log_dir, "resource_usage_plot.png")
            try:
                self.resource_monitor.plot_resource_usage(output_file=plot_path)
                logger.info(f"Resource usage plot saved to {plot_path}")
            except Exception as e:
                logger.error(f"Could not generate resource plot: {e}")
            
            # Get resource stats and stop monitoring
            stats = self.resource_monitor.get_usage_stats()
            self.resource_monitor.stop()
            
            # Log resource statistics
            logger.info(f"Average CPU usage: {stats['cpu_percent']['mean']:.1f}%")
            logger.info(f"Peak CPU usage: {stats['cpu_percent']['max']:.1f}%")
            logger.info(f"Average memory usage: {stats['memory_percent']['mean']:.1f}%")
            logger.info(f"Peak memory usage: {stats['memory_percent']['max']:.1f}%")
            
            if stats['gpu_utilization'] is not None:
                logger.info(f"Average GPU usage: {stats['gpu_utilization']['mean']:.1f}%")
                logger.info(f"Peak GPU usage: {stats['gpu_utilization']['max']:.1f}%")
        
        # Save final results
        results_path = os.path.join(self.checkpoint_dir, "hpo_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_params": best_params,
                "best_value": float(best_value),
                "n_trials": self.n_trials,
                "strategy": self.strategy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        return best_params, best_value
    
    def _run_optuna_optimization(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization with Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available. Install with 'pip install optuna'")
            raise ImportError("Optuna is required for this optimization strategy")
        
        # Set up sampler based on strategy
        if self.strategy == "random":
            sampler = RandomSampler(seed=self.seed)
        elif self.strategy == "cmaes":
            sampler = CmaEsSampler(seed=self.seed)
        else:
            sampler = TPESampler(seed=self.seed)
        
        # Set up pruner for early stopping
        pruner = MedianPruner() if self.early_stopping else None
        
        # Create study
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize"
        )
        
        # Prepare objective wrapper
        def objective_wrapper(trial):
            # Generate parameters from param_space
            params = {}
            for name, space_def in self.param_space.items():
                space_type, *space_args = space_def
                
                if space_type == "categorical":
                    params[name] = trial.suggest_categorical(name, space_args[0])
                elif space_type == "int":
                    params[name] = trial.suggest_int(name, space_args[0], space_args[1])
                elif space_type == "uniform":
                    params[name] = trial.suggest_float(name, space_args[0], space_args[1])
                elif space_type == "log_uniform":
                    params[name] = trial.suggest_float(name, space_args[0], space_args[1], log=True)
                else:
                    # Fixed parameter
                    params[name] = space_args[0]
            
            # Check if we should continue based on resource usage
            if not self._check_resources():
                raise optuna.TrialPruned("Resource limit exceeded")
            
            # Get trial_id for reporting
            trial_id = trial.number
            
            # Call the actual objective function
            try:
                value = self.objective_function(params, trial_id)
                
                # Update best if this is better
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = copy.deepcopy(params)
                    logger.info(f"Trial {trial_id}: New best value = {value:.6f}")
                else:
                    logger.info(f"Trial {trial_id}: Value = {value:.6f}")
                
                # Save checkpoint at interval
                if trial_id % self.checkpoint_interval == 0:
                    self._save_checkpoint(study.trials)
                
                return value
                
            except EarlyStoppingSignal:
                logger.info(f"Trial {trial_id} stopped early")
                raise optuna.TrialPruned("Early stopping")
            
            except Exception as e:
                logger.error(f"Error in trial {trial_id}: {e}")
                return float('-inf')
        
        # Run optimization
        try:
            study.optimize(
                objective_wrapper, 
                n_trials=self.n_trials,
                n_jobs=self.n_parallel if self.n_parallel > 1 else 1,
                timeout=None
            )
        except KeyboardInterrupt:
            logger.info("Optimization stopped by user")
        
        # Get best parameters and value
        best_params = study.best_params if study.best_trial else self.best_params
        best_value = study.best_value if study.best_trial else self.best_value
        
        # Save importance information if available
        try:
            importances = optuna.importance.get_param_importances(study)
            importances_path = os.path.join(self.checkpoint_dir, "param_importances.json")
            with open(importances_path, "w") as f:
                json.dump({k: float(v) for k, v in importances.items()}, f, indent=2)
        except:
            pass
        
        return best_params, best_value
    
    def _run_evolutionary_optimization(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization with evolutionary algorithm (DEAP)."""
        if not DEAP_AVAILABLE:
            logger.error("DEAP not available. Install with 'pip install deap'")
            raise ImportError("DEAP is required for evolutionary optimization")
            
        import deap.base
        import deap.creator
        import deap.tools
        import deap.algorithms
        
        # Create fitness class that maximizes the objective
        try:
            deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
            deap.creator.create("Individual", dict, fitness=deap.creator.FitnessMax)
        except RuntimeError:
            # Already created, can happen when called multiple times
            pass
        
        # Create toolbox
        toolbox = deap.base.Toolbox()
        
        # Register parameter generators
        for name, space_def in self.param_space.items():
            space_type, *space_args = space_def
            
            if space_type == "categorical":
                toolbox.register(f"attr_{name}", random.choice, space_args[0])
            elif space_type == "int":
                toolbox.register(f"attr_{name}", random.randint, space_args[0], space_args[1])
            elif space_type == "uniform":
                toolbox.register(f"attr_{name}", random.uniform, space_args[0], space_args[1])
            elif space_type == "log_uniform":
                def sample_log_uniform(low, high):
                    return np.exp(random.uniform(np.log(low), np.log(high)))
                toolbox.register(f"attr_{name}", sample_log_uniform, space_args[0], space_args[1])
            else:
                # Fixed parameter
                toolbox.register(f"attr_{name}", lambda x: x, space_args[0])
        
        # Register individual and population
        def create_individual():
            return deap.creator.Individual(
                {name: getattr(toolbox, f"attr_{name}")() for name in self.param_space.keys()}
            )
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        
        # Evaluation function
        def eval_individual(individual):
            if not self._check_resources():
                return (-float("inf"),)
                
            try:
                # Use the individual's trial number as trial_id
                if hasattr(individual, "trial_id"):
                    trial_id = individual.trial_id
                else:
                    individual.trial_id = len(self.results)
                    trial_id = individual.trial_id
                
                value = self.objective_function(individual, trial_id)
                
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = copy.deepcopy(individual)
                    logger.info(f"Trial {trial_id}: New best value = {value:.6f}")
                else:
                    logger.info(f"Trial {trial_id}: Value = {value:.6f}")
                
                return (value,)
            
            except EarlyStoppingSignal:
                logger.info(f"Trial {getattr(individual, 'trial_id', '?')} stopped early")
                return (0.0,)
            
            except Exception as e:
                logger.error(f"Error in trial {getattr(individual, 'trial_id', '?')}: {e}")
                return (-float("inf"),)
        
        toolbox.register("evaluate", eval_individual)
        
        # Genetic operators
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        
        # Create initial population
        population = toolbox.population(n=min(self.n_trials, 50))
        
        # Parameters for the algorithm
        cxpb, mutpb, ngen = 0.5, 0.2, (self.n_trials // len(population)) + 1
        
        # Add trial_id to individuals
        for i, ind in enumerate(population):
            ind.trial_id = i
        
        # Initialize stats tracking
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            # Run the evolutionary algorithm
            if self.n_parallel > 1:
                pool = multiprocessing.Pool(processes=self.n_parallel)
                toolbox.register("map", pool.map)
            
            # Run for specified generations or until we've used up trial budget
            trials_used = 0
            for gen in range(ngen):
                if trials_used >= self.n_trials:
                    break
                
                # Evaluate all individuals with invalid fitness
                invalid_ind = [ind for ind in population if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    trials_used += 1
                
                # Get statistics
                record = stats.compile(population)
                logger.info(f"Generation {gen+1}: {record}")
                
                # Save checkpoint at interval
                if gen % self.checkpoint_interval == 0:
                    self._save_checkpoint(population)
                
                # Check if we're out of trials
                if trials_used >= self.n_trials:
                    break
                
                # Select next generation and apply genetic operators
                offspring = toolbox.select(population, len(population))
                offspring = deap.algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
                
                # Set trial_id for new individuals
                for ind in offspring:
                    if not hasattr(ind, "trial_id"):
                        ind.trial_id = trials_used
                        trials_used += 1
                
                # Replace population
                population[:] = offspring
            
            if self.n_parallel > 1:
                pool.close()
                pool.join()
                
        except KeyboardInterrupt:
            logger.info("Optimization stopped by user")
        
        # Find best individual
        best_ind = deap.tools.selBest(population, 1)[0]
        
        return dict(best_ind), best_ind.fitness.values[0]
    
    def _crossover(self, ind1, ind2):
        """Custom crossover operator for hyperparameter dictionaries."""
        child1, child2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
        
        # Randomly select parameters to crossover
        for name in self.param_space:
            if random.random() < 0.5:
                child1[name], child2[name] = child2[name], child1[name]
        
        return child1, child2
    
    def _mutate(self, individual):
        """Custom mutation operator for hyperparameter dictionaries."""
        # Select a random parameter to mutate
        name = random.choice(list(self.param_space.keys()))
        space_def = self.param_space[name]
        space_type, *space_args = space_def
        
        # Apply mutation based on parameter type
        if space_type == "categorical":
            individual[name] = random.choice(space_args[0])
        elif space_type == "int":
            low, high = space_args[0], space_args[1]
            individual[name] = random.randint(low, high)
        elif space_type == "uniform":
            low, high = space_args[0], space_args[1]
            individual[name] = random.uniform(low, high)
        elif space_type == "log_uniform":
            low, high = space_args[0], space_args[1]
            individual[name] = np.exp(random.uniform(np.log(low), np.log(high)))
        
        return individual,
    
    def _run_grid_search(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization with grid search."""
        # Generate grid points for each parameter
        grid_points = {}
        for name, space_def in self.param_space.items():
            space_type, *space_args = space_def
            
            if space_type == "categorical":
                grid_points[name] = space_args[0]
            elif space_type == "int":
                low, high = space_args[0], space_args[1]
                # Limit points for large ranges
                step = max(1, (high - low) // 10) if high - low > 10 else 1
                grid_points[name] = list(range(low, high + 1, step))
            elif space_type == "uniform" or space_type == "log_uniform":
                low, high = space_args[0], space_args[1]
                # Generate 5 points for continuous ranges
                if space_type == "log_uniform":
                    grid_points[name] = np.exp(np.linspace(np.log(low), np.log(high), 5)).tolist()
                else:
                    grid_points[name] = np.linspace(low, high, 5).tolist()
            else:
                grid_points[name] = [space_args[0]]
        
        # Generate all combinations
        import itertools
        param_names = list(grid_points.keys())
        param_values = list(grid_points.values())
        combinations = list(itertools.product(*param_values))
        
        # Limit to n_trials
        if len(combinations) > self.n_trials:
            logger.warning(f"Grid has {len(combinations)} points, limiting to {self.n_trials} trials")
            random.seed(self.seed)
            random.shuffle(combinations)
            combinations = combinations[:self.n_trials]
        
        logger.info(f"Grid search with {len(combinations)} combinations")
        
        results = []
        best_value = float('-inf')
        best_params = None
        
        # Run trials
        if self.n_parallel > 1:
            # Process in batches to allow checkpointing
            batch_size = min(100, len(combinations))
            batches = [combinations[i:i+batch_size] for i in range(0, len(combinations), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                with multiprocessing.Pool(processes=self.n_parallel) as pool:
                    batch_params = [dict(zip(param_names, combo)) for combo in batch]
                    
                    # Create a partial function for the trials
                    from functools import partial
                    trial_func = partial(self._run_trial, param_names=param_names, start_idx=batch_idx*batch_size)
                    
                    # Execute in parallel
                    batch_results = pool.map(trial_func, batch_params)
                    
                    # Process batch results
                    for params, value in batch_results:
                        results.append((params, value))
                        
                        if value > best_value:
                            best_value = value
                            best_params = params
                            logger.info(f"New best value: {best_value:.6f}")
                    
                    # Save checkpoint
                    self._save_checkpoint(results)
                    
                    # Check resources
                    if not self._check_resources():
                        logger.warning("Stopping grid search due to resource constraints")
                        break
        else:
            # Sequential execution
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                
                # Check resources
                if not self._check_resources():
                    logger.warning("Stopping grid search due to resource constraints")
                    break
                
                try:
                    value = self.objective_function(params, i)
                    results.append((params, value))
                    
                    if value > best_value:
                        best_value = value
                        best_params = params
                        logger.info(f"Trial {i}: New best value = {value:.6f}")
                    else:
                        logger.info(f"Trial {i}: Value = {value:.6f}")
                        
                except EarlyStoppingSignal:
                    logger.info(f"Trial {i} stopped early")
                    results.append((params, float('-inf')))
                    
                except Exception as e:
                    logger.error(f"Error in trial {i}: {e}")
                    results.append((params, float('-inf')))
                
                # Save checkpoint at interval
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(results)
        
        return best_params, best_value
    
    def _run_trial(self, params, param_names, start_idx):
        """Execute a single trial for parallel processing."""
        trial_id = start_idx + param_names.index(params)
        
        try:
            value = self.objective_function(params, trial_id)
            return params, value
        except EarlyStoppingSignal:
            return params, float('-inf')
        except Exception as e:
            logger.error(f"Error in trial {trial_id}: {e}")
            return params, float('-inf')
    
    def _run_population_based_training(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization with Ray Tune's Population-Based Training."""
        if not RAY_AVAILABLE:
            logger.error("Ray not available. Install with 'pip install ray[tune]'")
            raise ImportError("Ray is required for PBT")
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        
        # Define search space for Ray Tune
        def sample_params(spec):
            params = {}
            for name, space_def in self.param_space.items():
                space_type, *space_args = space_def
                
                if space_type == "categorical":
                    params[name] = random.choice(space_args[0])
                elif space_type == "int":
                    params[name] = random.randint(space_args[0], space_args[1])
                elif space_type == "uniform":
                    params[name] = random.uniform(space_args[0], space_args[1])
                elif space_type == "log_uniform":
                    params[name] = np.exp(random.uniform(np.log(space_args[0]), np.log(space_args[1])))
                else:
                    params[name] = space_args[0]
            return params
        
        # Define PBT mutation operations
        def explore(config):
            # PBT mutation strategy
            result = copy.deepcopy(config)
            for name, space_def in self.param_space.items():
                space_type, *space_args = space_def
                
                # Skip categorical parameters
                if space_type in ["categorical", "fixed"]:
                    continue
                
                # Perturb numerical parameters
                if random.random() < 0.2:  # 20% chance to mutate
                    if space_type == "int":
                        low, high = space_args[0], space_args[1]
                        # Perturb by up to ±20% of range
                        delta = max(1, int((high - low) * 0.2))
                        result[name] = max(low, min(high, result[name] + random.randint(-delta, delta)))
                    else:
                        # For continuous params, perturb by factor of 0.8-1.2
                        factor = random.uniform(0.8, 1.2)
                        if space_type == "log_uniform":
                            # Ensure we stay in range
                            low, high = space_args[0], space_args[1]
                            result[name] = max(low, min(high, result[name] * factor))
                        else:
                            low, high = space_args[0], space_args[1]
                            result[name] = max(low, min(high, result[name] * factor))
                            
            return result
        
        # Define training function for Ray
        @ray.remote
        def train_with_config(config, trial_id):
            try:
                return self.objective_function(config, trial_id)
            except EarlyStoppingSignal:
                return float('-inf')
            except Exception as e:
                logger.error(f"Error in trial {trial_id}: {e}")
                return float('-inf')
        
        # Define PBT scheduler
        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="value",
            mode="max",
            perturbation_interval=1,  # How many iterations before perturbing
            hyperparam_mutations=self.param_space,  # Use our param space
            custom_explore_fn=explore
        )
        
        # Run PBT with tune.run
        pop_size = min(self.n_trials // 2, 10)  # Population size
        iterations = max(2, self.n_trials // pop_size)  # How many iterations to run
        
        logger.info(f"Running PBT with population {pop_size}, iterations {iterations}")
        
        # Define trainable function for Ray Tune
        def trainable(config, checkpoint_dir=None):
            for i in range(iterations):
                # Get trial_id from Ray
                trial_id = tune.get_trial_id()
                
                # Call objective function
                try:
                    value = self.objective_function(config, f"{trial_id}_{i}")
                    
                    # Report to Ray
                    tune.report(value=value, training_iteration=i+1)
                    
                    # Update our local best
                    if value > self.best_value:
                        self.best_value = value
                        self.best_params = copy.deepcopy(config)
                        logger.info(f"Trial {trial_id}_{i}: New best value = {value:.6f}")
                    
                except EarlyStoppingSignal:
                    # Report a bad value if early stopped
                    tune.report(value=float('-inf'), training_iteration=i+1)
                    break
                    
                except Exception as e:
                    logger.error(f"Error in trial {trial_id}_{i}: {e}")
                    tune.report(value=float('-inf'), training_iteration=i+1)
                    break
        
        # Run optimization
        analysis = tune.run(
            trainable,
            num_samples=pop_size,
            config={
                # Parameters will be sampled by Ray
                **{k: tune.sample_from(lambda _: sample_params(_)[k]) for k in self.param_space}
            },
            scheduler=pbt,
            resources_per_trial={"cpu": 1, "gpu": 0},
            local_dir=os.path.join(self.checkpoint_dir, "ray"),
            verbose=1,
            max_failures=2,
            checkpoint_freq=1,
            checkpoint_at_end=True
        )
        
        # Get best config
        try:
            best_trial = analysis.get_best_trial("value", "max", "last")
            ray_best_config = best_trial.config
            ray_best_value = best_trial.last_result["value"]
            
            # Check if Ray's best is better than our tracked best
            if ray_best_value > self.best_value:
                self.best_value = ray_best_value
                self.best_params = ray_best_config
                
            # Save all trial data
            analysis_path = os.path.join(self.checkpoint_dir, "ray_analysis.json")
            with open(analysis_path, "w") as f:
                json.dump(analysis.stats(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing Ray results: {e}")
        
        return self.best_params, self.best_value
    

if __name__ == "__main__":
    # Simple test
    def test_objective(params):
        x = params["x"]
        y = params["y"]
        # Rosenbrock function
        return -(100 * (y - x**2)**2 + (1 - x)**2)
    
    config = HPOConfig(
        param_space={
            "x": {"low": -2.0, "high": 2.0},
            "y": {"low": -2.0, "high": 2.0},
        },
        n_trials=50,
        n_parallel=2,
        optimization_method="optuna",
        early_stopping=True,
        patience=10,
        checkpoint_interval=5
    )
    
    hpo = HPEngine(config)
    result = hpo.optimize(test_objective)
    
    print(f"Best parameters: {result['best_params']}")
    print(f"Best value: {result['best_value']}")
    print(f"Total duration: {result['total_duration']:.2f} seconds")