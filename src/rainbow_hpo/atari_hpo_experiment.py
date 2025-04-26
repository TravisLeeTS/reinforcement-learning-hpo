"""
End-to-end hyperparameter optimization experiment for Rainbow DQN on Atari environments.
Implements multiple HPO strategies and comprehensive analysis of their performance.
"""

import os
import time
import json
import argparse
import logging
import random
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gymnasium as gym
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from datetime import datetime
from tqdm import tqdm
import optuna
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import concurrent.futures
import deap
from deap import base, creator, tools, algorithms

from agent_builder import RainbowDQNAgent, AgentBuilder
from env_builder import EnvironmentBuilder
from hpo_engine import HyperparameterOptimizer
from analyzer import Analyzer

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/atari_hpo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AtariExperimentConfig:
    """Configuration for the Atari HPO experiment."""
    
    def __init__(
        self,
        env_id: str = "PongNoFrameskip-v4",
        n_trials: int = 10,
        n_seeds: int = 3,
        budget_per_trial: int = 500000,  # steps per trial
        random_seed: int = 42,
        compute_resources: Dict[str, Any] = None,
        output_dir: str = "results"
    ):
        """
        Initialize the experiment configuration.
        
        Args:
            env_id: Atari environment ID
            n_trials: Number of trials per HPO method
            n_seeds: Number of seeds to run for statistical reliability
            budget_per_trial: Training budget per trial in steps
            random_seed: Master random seed
            compute_resources: Dictionary specifying compute resources
            output_dir: Directory to save results
        """
        self.env_id = env_id
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.budget_per_trial = budget_per_trial
        self.random_seed = random_seed
        self.output_dir = output_dir
        
        # Default compute resources based on available hardware
        default_resources = {
            "n_workers": min(os.cpu_count() or 1, 4),
            "use_gpu": torch.cuda.is_available(),
            "gpu_per_trial": 0.5 if torch.cuda.is_available() else 0,
            "cpu_per_trial": 1
        }
        
        self.compute_resources = compute_resources or default_resources
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log the configuration
        logger.info(f"Experiment Configuration:")
        logger.info(f"  Environment: {self.env_id}")
        logger.info(f"  Trials: {self.n_trials}")
        logger.info(f"  Seeds: {self.n_seeds}")
        logger.info(f"  Budget per trial: {self.budget_per_trial} steps")
        logger.info(f"  Random seed: {self.random_seed}")
        logger.info(f"  Compute resources: {self.compute_resources}")
        logger.info(f"  Output directory: {self.output_dir}")


class BaseHPOStrategy:
    """Base class for HPO strategies."""
    
    def __init__(
        self,
        config: AtariExperimentConfig,
        name: str
    ):
        """
        Initialize the HPO strategy.
        
        Args:
            config: Experiment configuration
            name: Name of the strategy
        """
        self.config = config
        self.name = name
        self.results_dir = os.path.join(config.output_dir, name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.best_params = None
        self.best_score = float('-inf')
        self.all_results = []
        self.runtime = 0
        
        # Set random seeds for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        
        logger.info(f"Initialized {name} HPO strategy")
    
    def get_param_space(self) -> Dict[str, Any]:
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary defining the hyperparameter search space
        """
        param_space = {
            "learning_rate": ("log_uniform", 1e-5, 1e-3),
            "gamma": ("uniform", 0.95, 0.999),
            "batch_size": ("categorical", [32, 64, 128, 256]),
            "target_update_freq": ("int", 500, 10000),
            "n_steps": ("int", 1, 5),
            "epsilon_start": ("uniform", 0.5, 1.0),
            "epsilon_end": ("uniform", 0.01, 0.1),
            "epsilon_decay_steps": ("int", 50000, 500000),
            "memory_size": ("categorical", [50000, 100000, 200000]),
            "n_atoms": ("categorical", [21, 51, 101]),
            "v_min": ("uniform", -20.0, -5.0),
            "v_max": ("uniform", 5.0, 20.0)
        }
        return param_space
    
    def evaluate_params(self, params: Dict[str, Any], seed: int, budget: int) -> float:
        """
        Evaluate a set of hyperparameters.
        
        Args:
            params: Hyperparameters to evaluate
            seed: Random seed
            budget: Training budget in steps
            
        Returns:
            Mean episode reward
        """
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env_builder = EnvironmentBuilder()
        train_env = env_builder.build_env(
            self.config.env_id,
            render_mode=None
        )
        eval_env = env_builder.build_env(
            self.config.env_id,
            render_mode=None
        )
        
        # Create agent
        agent_builder = AgentBuilder()
        agent = agent_builder.build_agent(
            train_env.observation_space,
            train_env.action_space,
            params,
            seed=seed
        )
        
        # Train agent
        episode_rewards, training_losses = [], []
        state, _ = train_env.reset(seed=seed)
        episode_reward = 0
        episode_steps = 0
        episode_count = 0
        
        # Metrics
        eval_freq = 10000  # steps between evaluations
        next_eval_at = eval_freq
        evaluation_rewards = []
        steps_to_threshold = None
        reward_threshold = 18.0  # for Pong, we consider solving at 18 points
        
        start_time = time.time()
        total_steps = 0
        
        # Keep track of resources
        process = psutil.Process(os.getpid())
        max_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Progress bar
        pbar = tqdm(total=budget, desc=f"Training (seed={seed})", leave=False)
        
        while total_steps < budget:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Optimize (train)
            loss = agent.optimize()
            if loss != 0:  # Only track non-zero losses
                training_losses.append(loss)
            
            episode_reward += reward
            state = next_state
            episode_steps += 1
            total_steps += 1
            pbar.update(1)
            
            # End of episode
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_steps = 0
                episode_count += 1
                state, _ = train_env.reset()
            
            # Periodic evaluation
            if total_steps >= next_eval_at:
                eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=5)
                evaluation_rewards.append(eval_reward)
                
                # Check if environment is solved
                if steps_to_threshold is None and eval_reward >= reward_threshold:
                    steps_to_threshold = total_steps
                
                next_eval_at += eval_freq
                
                # Update progress bar description
                pbar.set_description(f"Training (seed={seed}, eval={eval_reward:.2f})")
                
                # Update memory usage
                current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                max_memory_usage = max(max_memory_usage, current_memory)
        
        pbar.close()
        
        # Final evaluation
        final_eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=20)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Prepare metrics
        metrics = {
            "mean_reward": final_eval_reward,
            "training_time": training_time,
            "episodes_completed": episode_count,
            "steps_to_threshold": steps_to_threshold,
            "max_memory_mb": max_memory_usage,
            "evaluation_rewards": evaluation_rewards,
            "episode_rewards": episode_rewards[-100:] if episode_rewards else []  # Last 100 episodes
        }
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return final_eval_reward, metrics
    
    def _evaluate_agent(self, agent: RainbowDQNAgent, env: gym.Env, n_episodes: int = 10) -> float:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            agent: Rainbow DQN agent
            env: Gym environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Mean episode reward
        """
        episode_rewards = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        return mean_reward
    
    def save_results(self) -> None:
        """Save results to file."""
        results = {
            "method": self.name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "runtime": self.runtime,
            "total_trials": len(self.all_results),
            "all_results": self.all_results,
            "env_id": self.config.env_id,
            "n_trials": self.config.n_trials,
            "n_seeds": self.config.n_seeds,
            "budget_per_trial": self.config.budget_per_trial,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(self.results_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {results_file}")
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run the HPO strategy.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        raise NotImplementedError("Subclasses must implement run()")


class BayesianOptimizationStrategy(BaseHPOStrategy):
    """Bayesian Optimization strategy using Optuna."""
    
    def __init__(
        self,
        config: AtariExperimentConfig
    ):
        """
        Initialize the Bayesian Optimization strategy.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, "BayesianOptimization")
    
    def _create_optuna_objective(self, trial_idx: int) -> Callable:
        """
        Create an Optuna objective function.
        
        Args:
            trial_idx: Trial index
            
        Returns:
            Objective function
        """
        def objective(trial):
            # Define hyperparameter space
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "target_update_freq": trial.suggest_int("target_update_freq", 500, 10000, log=True),
                "n_steps": trial.suggest_int("n_steps", 1, 5),
                "epsilon_start": trial.suggest_float("epsilon_start", 0.5, 1.0),
                "epsilon_end": trial.suggest_float("epsilon_end", 0.01, 0.1),
                "epsilon_decay_steps": trial.suggest_int("epsilon_decay_steps", 50000, 500000, log=True),
                "memory_size": trial.suggest_categorical("memory_size", [50000, 100000, 200000]),
                "n_atoms": trial.suggest_categorical("n_atoms", [21, 51, 101]),
                "v_min": trial.suggest_float("v_min", -20.0, -5.0),
                "v_max": trial.suggest_float("v_max", 5.0, 20.0)
            }
            
            # Evaluate with multiple seeds
            all_rewards = []
            trial_metrics = []
            
            for seed_offset in range(self.config.n_seeds):
                seed = self.config.random_seed + seed_offset
                reward, metrics = self.evaluate_params(
                    params, 
                    seed=seed,
                    budget=self.config.budget_per_trial
                )
                all_rewards.append(reward)
                trial_metrics.append(metrics)
            
            # Calculate mean reward and other aggregate metrics
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            variance = np.var(all_rewards)
            
            # Calculate steps to threshold (if any seed reached it)
            steps_to_threshold_values = [m["steps_to_threshold"] for m in trial_metrics if m["steps_to_threshold"] is not None]
            mean_steps_to_threshold = np.mean(steps_to_threshold_values) if steps_to_threshold_values else None
            
            # Aggregate metrics
            aggregate_metrics = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "variance": variance,
                "rewards_per_seed": all_rewards,
                "mean_steps_to_threshold": mean_steps_to_threshold,
            }
            
            # Store result
            result = {
                "trial_id": trial_idx,
                "hyperparams": params,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "detailed_metrics": {
                    "per_seed": trial_metrics,
                    "aggregate": aggregate_metrics
                }
            }
            self.all_results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            return mean_reward
        
        return objective
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run Bayesian Optimization.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Running Bayesian Optimization for {self.config.n_trials} trials")
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
                interval_steps=1
            )
        )
        
        # Run trials
        for trial_idx in range(self.config.n_trials):
            objective = self._create_optuna_objective(trial_idx)
            study.optimize(objective, n_trials=1)
        
        # Calculate runtime
        self.runtime = time.time() - start_time
        
        # Get best params and score
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Calculate parameter importances
        param_importances = {}
        try:
            importances = optuna.importance.get_param_importances(study)
            param_importances = dict(importances)
        except Exception as e:
            logger.error(f"Error calculating parameter importances: {e}")
        
        # Save results
        results = {
            "method": self.name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "runtime": self.runtime,
            "total_trials": len(self.all_results),
            "all_results": self.all_results,
            "param_importances": param_importances,
            "env_id": self.config.env_id,
            "n_trials": self.config.n_trials,
            "n_seeds": self.config.n_seeds,
            "budget_per_trial": self.config.budget_per_trial,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(self.results_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Bayesian Optimization completed in {self.runtime:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.best_score


class EvolutionaryStrategy(BaseHPOStrategy):
    """Evolutionary Algorithm strategy using DEAP."""
    
    def __init__(
        self,
        config: AtariExperimentConfig
    ):
        """
        Initialize the Evolutionary Algorithm strategy.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, "EvolutionaryAlgorithm")
        
        # DEAP setup for maximizing fitness
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.param_bounds = {}
        self.param_types = {}
        self.categorical_options = {}
        self.gen_stats = []
    
    def _setup_deap(self):
        """Set up DEAP for the evolutionary algorithm."""
        param_space = self.get_param_space()
        
        # Initialize parameter bounds and types
        for param_name, param_config in param_space.items():
            dist_type = param_config[0]
            
            if dist_type in ["uniform", "log_uniform"]:
                min_val, max_val = param_config[1], param_config[2]
                self.param_bounds[param_name] = (min_val, max_val)
                self.param_types[param_name] = "float"
                
                # Register gene
                if dist_type == "uniform":
                    self.toolbox.register(f"attr_{param_name}", 
                                        random.uniform, min_val, max_val)
                else:  # log_uniform
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    self.toolbox.register(f"attr_{param_name}", 
                                        lambda: np.exp(random.uniform(log_min, log_max)))
            
            elif dist_type == "int":
                min_val, max_val = param_config[1], param_config[2]
                self.param_bounds[param_name] = (min_val, max_val)
                self.param_types[param_name] = "int"
                
                # Register gene
                self.toolbox.register(f"attr_{param_name}", 
                                    random.randint, min_val, max_val)
            
            elif dist_type == "categorical":
                options = param_config[1]
                self.categorical_options[param_name] = options
                self.param_types[param_name] = "categorical"
                
                # Register gene
                self.toolbox.register(f"attr_{param_name}", 
                                    random.choice, options)
        
        # Register individual creation and population
        param_names = list(param_space.keys())
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             [getattr(self.toolbox, f"attr_{name}") for name in param_names], n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _individual_to_params(self, individual) -> Dict[str, Any]:
        """
        Convert a DEAP individual to parameter dictionary.
        
        Args:
            individual: DEAP individual
            
        Returns:
            Dictionary of parameters
        """
        param_space = self.get_param_space()
        param_names = list(param_space.keys())
        params = {}
        
        for i, param_name in enumerate(param_names):
            value = individual[i]
            
            # Convert to correct type
            if self.param_types[param_name] == "int":
                value = int(value)
            elif self.param_types[param_name] == "float":
                value = float(value)
            
            params[param_name] = value
        
        return params
    
    def _evaluate_individual(self, individual) -> Tuple[float,]:
        """
        Evaluate a DEAP individual.
        
        Args:
            individual: DEAP individual
            
        Returns:
            Tuple containing fitness value
        """
        # Convert individual to parameters
        params = self._individual_to_params(individual)
        
        # Evaluate with multiple seeds
        all_rewards = []
        trial_metrics = []
        
        for seed_offset in range(self.config.n_seeds):
            seed = self.config.random_seed + seed_offset
            reward, metrics = self.evaluate_params(
                params, 
                seed=seed,
                budget=self.config.budget_per_trial
            )
            all_rewards.append(reward)
            trial_metrics.append(metrics)
        
        # Calculate mean reward and other aggregate metrics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        variance = np.var(all_rewards)
        
        # Calculate steps to threshold (if any seed reached it)
        steps_to_threshold_values = [m["steps_to_threshold"] for m in trial_metrics if m["steps_to_threshold"] is not None]
        mean_steps_to_threshold = np.mean(steps_to_threshold_values) if steps_to_threshold_values else None
        
        # Aggregate metrics
        aggregate_metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "variance": variance,
            "rewards_per_seed": all_rewards,
            "mean_steps_to_threshold": mean_steps_to_threshold,
        }
        
        # Store result
        result = {
            "trial_id": len(self.all_results),
            "hyperparams": params,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "detailed_metrics": {
                "per_seed": trial_metrics,
                "aggregate": aggregate_metrics
            }
        }
        self.all_results.append(result)
        
        # Save intermediate results
        self.save_results()
        
        return (mean_reward,)
    
    def _custom_crossover(self, ind1, ind2):
        """
        Custom crossover operator that respects parameter types.
        
        Args:
            ind1: First individual
            ind2: Second individual
            
        Returns:
            Tuple of offspring
        """
        param_space = self.get_param_space()
        param_names = list(param_space.keys())
        
        # Create copies of individuals
        offspring1, offspring2 = list(ind1), list(ind2)
        
        # Apply crossover with 70% probability
        if random.random() < 0.7:
            for i, param_name in enumerate(param_names):
                # 50% chance of swapping each parameter
                if random.random() < 0.5:
                    offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
        
        return creator.Individual(offspring1), creator.Individual(offspring2)
    
    def _custom_mutation(self, individual):
        """
        Custom mutation operator that respects parameter types and bounds.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Tuple containing the mutated individual
        """
        param_space = self.get_param_space()
        param_names = list(param_space.keys())
        
        for i, param_name in enumerate(param_names):
            # 20% chance of mutating each parameter
            if random.random() < 0.2:
                if self.param_types[param_name] == "float":
                    # Gaussian mutation for float parameters
                    min_val, max_val = self.param_bounds[param_name]
                    range_width = max_val - min_val
                    sigma = range_width * 0.1  # 10% of range as standard deviation
                    
                    # Apply mutation and clip to bounds
                    new_value = individual[i] + random.gauss(0, sigma)
                    new_value = max(min_val, min(max_val, new_value))
                    individual[i] = new_value
                
                elif self.param_types[param_name] == "int":
                    # Integer mutation
                    min_val, max_val = self.param_bounds[param_name]
                    range_width = max_val - min_val
                    sigma = max(1, int(range_width * 0.1))  # At least 1, or 10% of range
                    
                    # Apply mutation and clip to bounds
                    delta = random.randint(-sigma, sigma)
                    new_value = individual[i] + delta
                    new_value = max(min_val, min(max_val, new_value))
                    individual[i] = new_value
                
                elif self.param_types[param_name] == "categorical":
                    # Categorical mutation - select a different option
                    options = self.categorical_options[param_name]
                    current_value = individual[i]
                    other_options = [opt for opt in options if opt != current_value]
                    
                    if other_options:  # If there are other options available
                        individual[i] = random.choice(other_options)
        
        return (individual,)
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run Evolutionary Algorithm.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Running Evolutionary Algorithm for {self.config.n_trials} trials")
        start_time = time.time()
        
        # Set up DEAP
        self._setup_deap()
        
        # Create initial population
        pop_size = max(10, self.config.n_trials // 2)
        population = self.toolbox.population(n=pop_size)
        
        # Number of generations
        n_generations = self.config.n_trials // pop_size + 1
        
        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Run evolutionary algorithm
        pop, logbook = algorithms.eaSimple(
            population=population,
            toolbox=self.toolbox,
            cxpb=0.7,  # Probability of crossover
            mutpb=0.2,  # Probability of mutation
            ngen=n_generations,
            stats=stats,
            verbose=True
        )
        
        # Extract stats
        self.gen_stats = logbook.select("gen", "avg", "min", "max", "std")
        
        # Calculate runtime
        self.runtime = time.time() - start_time
        
        # Find best individual
        best_individual = tools.selBest(pop, k=1)[0]
        self.best_params = self._individual_to_params(best_individual)
        self.best_score = best_individual.fitness.values[0]
        
        # Save results
        results = {
            "method": self.name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "runtime": self.runtime,
            "total_trials": len(self.all_results),
            "all_results": self.all_results,
            "gen_stats": [
                {
                    "generation": gen,
                    "avg": avg,
                    "min": min_val,
                    "max": max_val,
                    "std": std_val
                }
                for gen, avg, min_val, max_val, std_val in zip(
                    self.gen_stats[0], self.gen_stats[1], 
                    self.gen_stats[2], self.gen_stats[3], self.gen_stats[4]
                )
            ],
            "env_id": self.config.env_id,
            "n_trials": self.config.n_trials,
            "n_seeds": self.config.n_seeds,
            "budget_per_trial": self.config.budget_per_trial,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(self.results_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Evolutionary Algorithm completed in {self.runtime:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.best_score


class PopulationBasedTrainingStrategy(BaseHPOStrategy):
    """Population-Based Training strategy using Ray Tune."""
    
    def __init__(
        self,
        config: AtariExperimentConfig
    ):
        """
        Initialize the Population-Based Training strategy.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config, "PopulationBasedTraining")
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    def _pbt_objective(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None) -> None:
        """
        Objective function for PBT.
        
        Args:
            config: Hyperparameter configuration
            checkpoint_dir: Directory containing checkpoint
        """
        # Extract trial info
        trial_id = tune.get_trial_id()
        seed = self.config.random_seed + int(trial_id.split("_")[-1])
        
        # Convert config to params
        params = {k: v for k, v in config.items() if k != "seed"}
        
        # Create environment
        env_builder = EnvironmentBuilder()
        train_env = env_builder.build_env(
            self.config.env_id,
            render_mode=None
        )
        eval_env = env_builder.build_env(
            self.config.env_id,
            render_mode=None
        )
        
        # Create agent
        agent_builder = AgentBuilder()
        agent = agent_builder.build_agent(
            train_env.observation_space,
            train_env.action_space,
            params,
            seed=seed
        )
        
        # Load checkpoint if available
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            agent.load(checkpoint_path)
        
        # Train agent for a fixed number of steps between tune reconfigurations
        steps_per_iter = 10000
        total_steps = 0
        max_steps = self.config.budget_per_trial
        
        # Metrics
        eval_freq = min(10000, steps_per_iter)  # steps between evaluations
        next_eval_at = eval_freq
        evaluation_rewards = []
        steps_to_threshold = None
        reward_threshold = 18.0  # for Pong, we consider solving at 18 points
        
        # Training loop until max steps
        while total_steps < max_steps:
            # Train for steps_per_iter or until max_steps
            steps_this_iter = min(steps_per_iter, max_steps - total_steps)
            if steps_this_iter <= 0:
                break
                
            # Reset environment for the iteration
            state, _ = train_env.reset(seed=seed)
            
            # Step counter
            steps_taken = 0
            episode_rewards = []
            episode_reward = 0
            
            while steps_taken < steps_this_iter:
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Optimize (train)
                agent.optimize()
                
                episode_reward += reward
                state = next_state
                steps_taken += 1
                total_steps += 1
                
                # End of episode
                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state, _ = train_env.reset()
                
                # Periodic evaluation
                if total_steps >= next_eval_at:
                    eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=5)
                    evaluation_rewards.append(eval_reward)
                    
                    # Check if environment is solved
                    if steps_to_threshold is None and eval_reward >= reward_threshold:
                        steps_to_threshold = total_steps
                    
                    next_eval_at += eval_freq
                    
                    # Report to Ray Tune
                    tune.report(
                        mean_reward=eval_reward,
                        total_steps=total_steps,
                        episodes_completed=len(episode_rewards),
                        steps_to_threshold=steps_to_threshold,
                    )
            
            # Save checkpoint for PBT to use
            checkpoint_path = os.path.join(tune.get_trial_dir(), "checkpoint.pth")
            agent.save(checkpoint_path)
        
        # Final evaluation
        final_eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=20)
        
        # Report final metrics
        tune.report(
            mean_reward=final_eval_reward,
            total_steps=total_steps,
            episodes_completed=len(episode_rewards),
            steps_to_threshold=steps_to_threshold,
            final_eval=True
        )
        
        # Clean up
        train_env.close()
        eval_env.close()
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run Population-Based Training.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Running Population-Based Training for {self.config.n_trials} trials")
        start_time = time.time()
        
        # Define hyperparameter space
        param_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "gamma": tune.uniform(0.95, 0.999),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "target_update_freq": tune.randint(500, 10000),
            "n_steps": tune.randint(1, 5),
            "epsilon_start": tune.uniform(0.5, 1.0),
            "epsilon_end": tune.uniform(0.01, 0.1),
            "epsilon_decay_steps": tune.randint(50000, 500000),
            "memory_size": tune.choice([50000, 100000, 200000]),
            "n_atoms": tune.choice([21, 51, 101]),
            "v_min": tune.uniform(-20.0, -5.0),
            "v_max": tune.uniform(5.0, 20.0),
            "seed": self.config.random_seed
        }
        
        # PBT scheduler
        pbt = PopulationBasedTraining(
            time_attr="total_steps",
            metric="mean_reward",
            mode="max",
            perturbation_interval=10000,  # Evaluate every 10K steps
            hyperparam_mutations={
                "learning_rate": lambda: random.uniform(1e-5, 1e-3),
                "gamma": lambda: random.uniform(0.95, 0.999),
                "target_update_freq": lambda: random.randint(500, 10000),
                "epsilon_start": lambda: random.uniform(0.5, 1.0),
                "epsilon_end": lambda: random.uniform(0.01, 0.1),
            }
        )
        
        # Run PBT
        population_size = min(self.config.n_trials, 10)  # Use a reasonable population size
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._pbt_objective),
                resources={"cpu": self.config.compute_resources["cpu_per_trial"], "gpu": self.config.compute_resources["gpu_per_trial"]}
            ),
            run_config=tune.RunConfig(
                name="pbt_rainbow",
                local_dir=self.results_dir,
                stop={"total_steps": self.config.budget_per_trial}
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=population_size
            ),
            param_space=param_space
        )
        
        results = tuner.fit()
        
        # Calculate runtime
        self.runtime = time.time() - start_time
        
        # Get best trial
        best_trial = results.get_best_trial("mean_reward", "max", scope="all")
        self.best_params = {k: v for k, v in best_trial.config.items() if k != "seed"}
        self.best_score = best_trial.last_result["mean_reward"]
        
        # Extract all results
        for i, trial in enumerate(results.get_all_trials()):
            # Process trial data
            trial_params = {k: v for k, v in trial.config.items() if k != "seed"}
            trial_history = trial.metric_analysis["mean_reward"]["history"]
            last_reward = trial.last_result.get("mean_reward", float('-inf'))
            steps_to_threshold = trial.last_result.get("steps_to_threshold", None)
            
            # Create detailed metrics
            detailed_metrics = {
                "reward_history": trial_history,
                "steps_to_threshold": steps_to_threshold,
                "final_reward": last_reward
            }
            
            # Store result
            result = {
                "trial_id": i,
                "hyperparams": trial_params,
                "mean_reward": last_reward,
                "detailed_metrics": detailed_metrics
            }
            self.all_results.append(result)
        
        # Save results
        results = {
            "method": self.name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "runtime": self.runtime,
            "total_trials": len(self.all_results),
            "all_results": self.all_results,
            "env_id": self.config.env_id,
            "n_trials": self.config.n_trials,
            "n_seeds": self.config.n_seeds,
            "budget_per_trial": self.config.budget_per_trial,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(self.results_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        # Shutdown Ray
        ray.shutdown()
        
        logger.info(f"Population-Based Training completed in {self.runtime:.2f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.best_score


def train_best_agent(
    best_params: Dict[str, Any],
    config: AtariExperimentConfig,
    output_dir: str,
    render: bool = True
) -> None:
    """
    Train an agent using the best found hyperparameters.
    
    Args:
        best_params: Best hyperparameters
        config: Experiment configuration
        output_dir: Output directory
        render: Whether to render the environment
    """
    logger.info(f"Training agent with best hyperparameters: {best_params}")
    
    # Create environment
    env_builder = EnvironmentBuilder()
    render_mode = "human" if render else None
    env = env_builder.build_env(
        config.env_id,
        render_mode=render_mode
    )
    
    # Create agent
    agent_builder = AgentBuilder()
    agent = agent_builder.build_agent(
        env.observation_space,
        env.action_space,
        best_params,
        seed=config.random_seed
    )
    
    # Training loop
    total_steps = 0
    max_steps = 1000000  # 1M steps for final agent training
    episodes = 0
    episode_rewards = []
    
    # Progress bar
    pbar = tqdm(total=max_steps, desc="Training best agent")
    
    # Save directory
    best_agent_dir = os.path.join(output_dir, "best_agent")
    os.makedirs(best_agent_dir, exist_ok=True)
    
    # Training loop
    state, _ = env.reset(seed=config.random_seed)
    episode_reward = 0
    
    # Save rewards for plotting
    eval_rewards = []
    eval_freq = 50000  # Evaluate every 50K steps
    next_eval_at = eval_freq
    
    while total_steps < max_steps:
        # Select action
        action = agent.select_action(state)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Optimize (train)
        agent.optimize()
        
        episode_reward += reward
        state = next_state
        total_steps += 1
        pbar.update(1)
        
        # End of episode
        if done:
            episode_rewards.append(episode_reward)
            episodes += 1
            
            # Log progress every 10 episodes
            if episodes % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                pbar.set_description(f"Training best agent | Avg reward: {avg_reward:.2f}")
            
            episode_reward = 0
            state, _ = env.reset()
        
        # Periodic saving
        if total_steps % 100000 == 0:
            checkpoint_path = os.path.join(best_agent_dir, f"checkpoint_{total_steps}.zip")
            agent.save(checkpoint_path)
        
        # Periodic evaluation
        if total_steps >= next_eval_at:
            # Create a separate environment for evaluation
            eval_env = env_builder.build_env(
                config.env_id,
                render_mode=None
            )
            
            eval_reward = 0
            for _ in range(10):  # 10 episodes for evaluation
                eval_state, _ = eval_env.reset()
                eval_done = False
                eval_episode_reward = 0
                
                while not eval_done:
                    eval_action = agent.select_action(eval_state, eval_mode=True)
                    eval_next_state, eval_rew, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                    eval_done = eval_terminated or eval_truncated
                    
                    eval_episode_reward += eval_rew
                    eval_state = eval_next_state
                
                eval_reward += eval_episode_reward
            
            eval_reward /= 10  # Average over 10 episodes
            eval_rewards.append((total_steps, eval_reward))
            next_eval_at += eval_freq
            
            eval_env.close()
    
    pbar.close()
    
    # Save final model
    final_path = os.path.join(best_agent_dir, "final_model.zip")
    agent.save(final_path)
    
    # Save training metrics
    episode_rewards_path = os.path.join(best_agent_dir, "episode_rewards.npy")
    eval_rewards_path = os.path.join(best_agent_dir, "eval_rewards.npy")
    
    np.save(episode_rewards_path, np.array(episode_rewards))
    np.save(eval_rewards_path, np.array(eval_rewards))
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f"Training Rewards ({config.env_id})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Plot evaluation rewards
    if eval_rewards:
        plt.subplot(1, 2, 2)
        steps, rewards = zip(*eval_rewards)
        plt.plot(steps, rewards, 'r-')
        plt.title("Evaluation Rewards")
        plt.xlabel("Steps")
        plt.ylabel("Avg. Reward (10 episodes)")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(best_agent_dir, "learning_curve.png"))
    
    logger.info(f"Best agent training completed. Trained for {total_steps} steps across {episodes} episodes.")
    logger.info(f"Final model saved to {final_path}")
    
    # Close environment
    env.close()


def run_experiment(args: argparse.Namespace) -> None:
    """
    Run the hyperparameter optimization experiment.
    
    Args:
        args: Command-line arguments
    """
    # Create experiment configuration
    config = AtariExperimentConfig(
        env_id=args.env,
        n_trials=args.trials,
        n_seeds=args.seeds,
        budget_per_trial=args.budget,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Create results directory for this experiment
    experiment_name = f"{config.env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(config.output_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment configuration
    config_file = os.path.join(results_dir, "experiment_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "env_id": config.env_id,
            "n_trials": config.n_trials,
            "n_seeds": config.n_seeds,
            "budget_per_trial": config.budget_per_trial,
            "random_seed": config.random_seed,
            "compute_resources": config.compute_resources,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name
        }, f, indent=4)
    
    # Run HPO strategies
    strategies = []
    best_params_per_strategy = {}
    
    if args.all or args.bayesian:
        # Bayesian Optimization
        bayesian = BayesianOptimizationStrategy(config)
        best_params, best_score = bayesian.run()
        best_params_per_strategy["bayesian"] = (best_params, best_score)
        strategies.append("bayesian")
    
    if args.all or args.evolutionary:
        # Evolutionary Algorithm
        evolutionary = EvolutionaryStrategy(config)
        best_params, best_score = evolutionary.run()
        best_params_per_strategy["evolutionary"] = (best_params, best_score)
        strategies.append("evolutionary")
    
    if args.all or args.pbt:
        # Population-Based Training
        pbt = PopulationBasedTrainingStrategy(config)
        best_params, best_score = pbt.run()
        best_params_per_strategy["pbt"] = (best_params, best_score)
        strategies.append("pbt")
    
    # Analyze and compare results
    analyzer = Analyzer(results_dir=results_dir)
    analyzer.load_results(strategies)
    
    # Generate comparison plots
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    analyzer.plot_learning_curves(save_path=os.path.join(figures_dir, "learning_curves.png"))
    analyzer.plot_method_comparison(save_path=os.path.join(figures_dir, "method_comparison.png"))
    analyzer.plot_hyperparameter_importance(save_path=os.path.join(figures_dir, "hyperparameter_importance.png"))
    analyzer.plot_reward_distributions(save_path=os.path.join(figures_dir, "reward_distributions.png"))
    
    # Generate comprehensive report
    report_path = analyzer.generate_report(output_dir=os.path.join(results_dir, "reports"))
    
    # Find overall best strategy
    best_strategy = None
    best_overall_score = float('-inf')
    
    for strategy, (params, score) in best_params_per_strategy.items():
        if score > best_overall_score:
            best_overall_score = score
            best_strategy = strategy
    
    if best_strategy:
        logger.info(f"Best overall strategy: {best_strategy} with score {best_overall_score:.4f}")
        
        # Train agent with best hyperparameters if requested
        if args.train_best:
            best_params = best_params_per_strategy[best_strategy][0]
            train_best_agent(best_params, config, results_dir, render=args.render)


def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(
        description="End-to-end hyperparameter optimization experiment for Rainbow DQN on Atari environments"
    )
    
    parser.add_argument(
        "--env", 
        type=str, 
        default="PongNoFrameskip-v4",
        help="Atari environment ID"
    )
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=20,
        help="Number of trials per HPO method"
    )
    
    parser.add_argument(
        "--seeds", 
        type=int, 
        default=3,
        help="Number of random seeds per trial for statistical reliability"
    )
    
    parser.add_argument(
        "--budget", 
        type=int, 
        default=500000,
        help="Training budget per trial in steps"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Master random seed"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    # HPO strategies
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all HPO strategies"
    )
    
    parser.add_argument(
        "--bayesian", 
        action="store_true",
        help="Run Bayesian Optimization"
    )
    
    parser.add_argument(
        "--evolutionary", 
        action="store_true",
        help="Run Evolutionary Algorithm"
    )
    
    parser.add_argument(
        "--pbt", 
        action="store_true",
        help="Run Population-Based Training"
    )
    
    parser.add_argument(
        "--train-best", 
        action="store_true",
        help="Train an agent using the best found hyperparameters"
    )
    
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render the environment when training the best agent"
    )
    
    args = parser.parse_args()
    
    # Default to --all if no strategy is specified
    if not (args.all or args.bayesian or args.evolutionary or args.pbt):
        args.all = True
    
    run_experiment(args)


if __name__ == "__main__":
    main()