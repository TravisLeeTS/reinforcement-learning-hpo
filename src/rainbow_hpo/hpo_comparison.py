"""
Hyperparameter Optimization Comparison for Rainbow DQN.

This script provides a holistic comparison between three HPO methods:
1. Bayesian Optimization (using Optuna)
2. Evolutionary Algorithm (using DEAP)
3. Population-Based Training (using Ray Tune)

The comparison includes:
- Performance metrics (mean reward, variance)
- Convergence speed (steps to threshold, episodes to convergence)
- Computational efficiency (runtime, memory usage)
- Optimal hyperparameters discovered

This code is designed to generate comprehensive data for thesis studies and academic research.
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
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# HPO libraries
import optuna
from deap import base, creator, tools, algorithms
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

# Import project modules
from agent_builder import RainbowDQNAgent, AgentBuilder
from env_builder import EnvironmentBuilder
from analyzer import Analyzer

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/hpo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HPOComparison:
    """Class for comparing different HPO methods."""
    
    def __init__(
        self,
        env_id: str = "PongNoFrameskip-v4",
        n_trials: int = 20,
        n_seeds: int = 3,
        budget_per_trial: int = 500000,
        random_seed: int = 42,
        output_dir: str = "comparison_results",
        early_stopping_threshold: Optional[float] = None,
        patience: int = 3
    ):
        """
        Initialize the HPO comparison.
        
        Args:
            env_id: Gym environment ID
            n_trials: Number of trials per HPO method
            n_seeds: Number of random seeds for statistical reliability
            budget_per_trial: Training budget per trial in environment steps
            random_seed: Master random seed for reproducibility
            output_dir: Directory to save comparison results
            early_stopping_threshold: Reward threshold for early stopping (None to disable)
            patience: Number of consecutive trials without improvement before stopping
        """
        self.env_id = env_id
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.budget_per_trial = budget_per_trial
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.early_stopping_threshold = early_stopping_threshold
        self.patience = patience
        
        # Create output directory
        self.base_dir = os.path.join(output_dir, f"{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create directories for each method
        self.bayesian_dir = os.path.join(self.base_dir, "bayesian")
        self.evolutionary_dir = os.path.join(self.base_dir, "evolutionary")
        self.pbt_dir = os.path.join(self.base_dir, "pbt")
        
        os.makedirs(self.bayesian_dir, exist_ok=True)
        os.makedirs(self.evolutionary_dir, exist_ok=True)
        os.makedirs(self.pbt_dir, exist_ok=True)
        
        # Store results
        self.results = {
            "bayesian": None,
            "evolutionary": None,
            "pbt": None
        }
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Save configuration
        self.config = {
            "env_id": env_id,
            "n_trials": n_trials,
            "n_seeds": n_seeds,
            "budget_per_trial": budget_per_trial,
            "random_seed": random_seed,
            "early_stopping_threshold": early_stopping_threshold,
            "patience": patience,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        logger.info(f"Initialized HPO comparison with environment: {env_id}")
        logger.info(f"Trials: {n_trials}, Seeds: {n_seeds}, Budget: {budget_per_trial}")
        if early_stopping_threshold is not None:
            logger.info(f"Early stopping enabled at threshold: {early_stopping_threshold} with patience: {patience}")
    
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
    
    def evaluate_params(self, params: Dict[str, Any], seed: int) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a set of hyperparameters.
        
        Args:
            params: Hyperparameters to evaluate
            seed: Random seed
            
        Returns:
            Tuple of (mean_reward, metrics)
        """
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env_builder = EnvironmentBuilder()
        train_env = env_builder.build_env(
            self.env_id,
            render_mode=None
        )
        eval_env = env_builder.build_env(
            self.env_id,
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
        
        # Training metrics
        episode_rewards = []
        training_losses = []
        evaluation_rewards = []
        steps_to_threshold = None
        reward_threshold = 18.0  # for Pong, we consider solving at 18 points
        
        # Reset environment
        state, _ = train_env.reset(seed=seed)
        episode_reward = 0
        episode_count = 0
        
        # Resource tracking
        start_time = time.time()
        process = psutil.Process(os.getpid())
        max_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Evaluation frequency
        eval_freq = 10000  # steps
        next_eval_at = eval_freq
        
        # Progress bar
        total_steps = 0
        with tqdm(total=self.budget_per_trial, desc=f"Training (seed={seed})", leave=False) as pbar:
            while total_steps < self.budget_per_trial:
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Optimize (train)
                loss = agent.optimize()
                if loss != 0:
                    training_losses.append(loss)
                
                episode_reward += reward
                state = next_state
                total_steps += 1
                
                # Update progress bar
                pbar.update(1)
                
                # End of episode
                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    episode_count += 1
                    state, _ = train_env.reset()
                
                # Periodic evaluation
                if total_steps >= next_eval_at:
                    eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=5)
                    evaluation_rewards.append((total_steps, eval_reward))
                    
                    # Check if environment is solved
                    if steps_to_threshold is None and eval_reward >= reward_threshold:
                        steps_to_threshold = total_steps
                    
                    next_eval_at += eval_freq
                    
                    # Update progress bar description
                    pbar.set_description(f"Training (seed={seed}, eval={eval_reward:.2f})")
                    
                    # Update memory usage
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    max_memory_usage = max(max_memory_usage, current_memory)
        
        # Final evaluation
        final_eval_reward = self._evaluate_agent(agent, eval_env, n_episodes=20)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Prepare metrics
        metrics = {
            "mean_reward": final_eval_reward,
            "training_time": training_time,
            "episodes_completed": episode_count,
            "max_memory_mb": max_memory_usage,
            "steps_to_threshold": steps_to_threshold,
            "evaluation_rewards": evaluation_rewards,
            "episode_rewards": episode_rewards[-100:] if episode_rewards else [],  # Last 100 episodes
            "mean_episode_length": total_steps / max(1, episode_count)
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
    
    def run_bayesian_optimization(self) -> Dict[str, Any]:
        """
        Run Bayesian Optimization using Optuna.
        
        Returns:
            Dictionary of results
        """
        logger.info(f"Starting Bayesian Optimization with {self.n_trials} trials")
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
                interval_steps=1
            )
        )
        
        # Store all trial results
        all_results = []
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
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
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_seed + seed_offset
                reward, metrics = self.evaluate_params(params, seed=seed)
                all_rewards.append(reward)
                trial_metrics.append(metrics)
            
            # Calculate mean and std of rewards
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            variance = np.var(all_rewards)
            
            # Calculate steps to threshold (if any seed reached it)
            steps_to_threshold_values = [m["steps_to_threshold"] for m in trial_metrics if m["steps_to_threshold"] is not None]
            mean_steps_to_threshold = np.mean(steps_to_threshold_values) if steps_to_threshold_values else None
            
            # Calculate mean training time
            mean_training_time = np.mean([m["training_time"] for m in trial_metrics])
            
            # Calculate mean memory usage
            mean_memory_usage = np.mean([m["max_memory_mb"] for m in trial_metrics])
            
            # Store trial result
            trial_result = {
                "trial_id": trial.number,
                "hyperparams": params,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "variance": variance,
                "mean_steps_to_threshold": mean_steps_to_threshold,
                "mean_training_time": mean_training_time,
                "mean_memory_usage": mean_memory_usage,
                "all_rewards": all_rewards,
                "detailed_metrics": {
                    "per_seed": trial_metrics,
                    "aggregate": {
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "variance": variance,
                        "steps_to_threshold": mean_steps_to_threshold,
                        "rewards_per_seed": all_rewards
                    }
                }
            }
            
            all_results.append(trial_result)
            
            return mean_reward
        
        # Run trials
        study.optimize(objective, n_trials=self.n_trials)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Calculate parameter importances
        param_importances = {}
        try:
            importances = optuna.importance.get_param_importances(study)
            param_importances = dict(importances)
        except Exception as e:
            logger.error(f"Error calculating parameter importances: {e}")
        
        # Get best params and score
        best_params = study.best_params
        best_score = study.best_value
        
        # Prepare results
        results = {
            "method": "Bayesian Optimization",
            "best_params": best_params,
            "best_score": best_score,
            "runtime": runtime,
            "total_trials": self.n_trials,
            "all_results": all_results,
            "param_importances": param_importances,
            "all_trials": [
                {
                    "trial_id": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        # Save results
        results_file = os.path.join(self.bayesian_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Bayesian Optimization completed in {runtime:.2f}s")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return results
    
    def run_evolutionary_algorithm(self) -> Dict[str, Any]:
        """
        Run Evolutionary Algorithm using DEAP.
        
        Returns:
            Dictionary of results
        """
        logger.info(f"Starting Evolutionary Algorithm with {self.n_trials} trials")
        start_time = time.time()
        
        # Set up DEAP
        # Create fitness and individual classes if they don't exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        param_space = self.get_param_space()
        
        # Parameter metadata
        param_bounds = {}
        param_types = {}
        categorical_options = {}
        
        # Set up parameter genes
        for param_name, param_config in param_space.items():
            dist_type = param_config[0]
            
            if dist_type in ["uniform", "log_uniform"]:
                min_val, max_val = param_config[1], param_config[2]
                param_bounds[param_name] = (min_val, max_val)
                param_types[param_name] = "float"
                
                # Register gene
                if dist_type == "uniform":
                    toolbox.register(f"attr_{param_name}", 
                                   random.uniform, min_val, max_val)
                else:  # log_uniform
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    toolbox.register(f"attr_{param_name}", 
                                   lambda: float(np.exp(random.uniform(log_min, log_max))))
            
            elif dist_type == "int":
                min_val, max_val = param_config[1], param_config[2]
                param_bounds[param_name] = (min_val, max_val)
                param_types[param_name] = "int"
                
                # Register gene
                toolbox.register(f"attr_{param_name}", 
                               random.randint, min_val, max_val)
            
            elif dist_type == "categorical":
                options = param_config[1]
                categorical_options[param_name] = options
                param_types[param_name] = "categorical"
                
                # Register gene
                toolbox.register(f"attr_{param_name}", 
                               random.choice, options)
        
        # Function to convert an individual to params dictionary
        param_names = list(param_space.keys())
        def individual_to_params(individual):
            params = {}
            for i, param_name in enumerate(param_names):
                value = individual[i]
                
                # Convert to correct type
                if param_types[param_name] == "int":
                    value = int(value)
                elif param_types[param_name] == "float":
                    value = float(value)
                
                params[param_name] = value
            
            return params
        
        # Store all results
        all_results = []
        
        # Evaluation function
        def evaluate_individual(individual):
            # Convert individual to parameters
            params = individual_to_params(individual)
            
            # Evaluate with multiple seeds
            all_rewards = []
            trial_metrics = []
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_seed + seed_offset
                reward, metrics = self.evaluate_params(params, seed=seed)
                all_rewards.append(reward)
                trial_metrics.append(metrics)
            
            # Calculate mean reward and other aggregate metrics
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            variance = np.var(all_rewards)
            
            # Calculate steps to threshold (if any seed reached it)
            steps_to_threshold_values = [m["steps_to_threshold"] for m in trial_metrics if m["steps_to_threshold"] is not None]
            mean_steps_to_threshold = np.mean(steps_to_threshold_values) if steps_to_threshold_values else None
            
            # Store trial result
            trial_result = {
                "trial_id": len(all_results),
                "hyperparams": params,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "variance": variance,
                "mean_steps_to_threshold": mean_steps_to_threshold,
                "all_rewards": all_rewards,
                "detailed_metrics": {
                    "per_seed": trial_metrics,
                    "aggregate": {
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "variance": variance,
                        "steps_to_threshold": mean_steps_to_threshold,
                        "rewards_per_seed": all_rewards
                    }
                }
            }
            
            all_results.append(trial_result)
            
            return (mean_reward,)
        
        # Custom crossover operator
        def custom_crossover(ind1, ind2):
            # Create copies of individuals
            offspring1, offspring2 = list(ind1), list(ind2)
            
            # Apply crossover with 70% probability
            if random.random() < 0.7:
                for i, param_name in enumerate(param_names):
                    # 50% chance of swapping each parameter
                    if random.random() < 0.5:
                        offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
            
            return creator.Individual(offspring1), creator.Individual(offspring2)
        
        # Custom mutation operator
        def custom_mutation(individual):
            for i, param_name in enumerate(param_names):
                # 20% chance of mutating each parameter
                if random.random() < 0.2:
                    if param_types[param_name] == "float":
                        # Gaussian mutation for float parameters
                        min_val, max_val = param_bounds[param_name]
                        range_width = max_val - min_val
                        sigma = range_width * 0.1  # 10% of range as standard deviation
                        
                        # Apply mutation and clip to bounds
                        new_value = individual[i] + random.gauss(0, sigma)
                        new_value = max(min_val, min(max_val, new_value))
                        individual[i] = new_value
                    
                    elif param_types[param_name] == "int":
                        # Integer mutation
                        min_val, max_val = param_bounds[param_name]
                        range_width = max_val - min_val
                        sigma = max(1, int(range_width * 0.1))  # At least 1, or 10% of range
                        
                        # Apply mutation and clip to bounds
                        delta = random.randint(-sigma, sigma)
                        new_value = individual[i] + delta
                        new_value = max(min_val, min(max_val, new_value))
                        individual[i] = new_value
                    
                    elif param_types[param_name] == "categorical":
                        # Categorical mutation - select a different option
                        options = categorical_options[param_name]
                        current_value = individual[i]
                        other_options = [opt for opt in options if opt != current_value]
                        
                        if other_options:  # If there are other options available
                            individual[i] = random.choice(other_options)
            
            return (individual,)
        
        # Register individual creation and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        [getattr(toolbox, f"attr_{name}") for name in param_names], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", custom_crossover)
        toolbox.register("mutate", custom_mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop_size = max(10, self.n_trials // 2)
        population = toolbox.population(n=pop_size)
        
        # Number of generations
        n_generations = self.n_trials // pop_size + 1
        
        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Run evolutionary algorithm
        pop, logbook = algorithms.eaSimple(
            population=population,
            toolbox=toolbox,
            cxpb=0.7,  # Probability of crossover
            mutpb=0.2,  # Probability of mutation
            ngen=n_generations,
            stats=stats,
            verbose=True
        )
        
        # Extract stats
        gen_stats = logbook.select("gen", "avg", "min", "max", "std")
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Find best individual
        best_individual = tools.selBest(pop, k=1)[0]
        best_params = individual_to_params(best_individual)
        best_score = best_individual.fitness.values[0]
        
        # Prepare results
        results = {
            "method": "Evolutionary Algorithm",
            "best_params": best_params,
            "best_score": best_score,
            "runtime": runtime,
            "total_trials": len(all_results),
            "all_results": all_results,
            "gen_stats": [
                {
                    "generation": gen,
                    "avg": avg,
                    "min": min_val,
                    "max": max_val,
                    "std": std_val
                }
                for gen, avg, min_val, max_val, std_val in zip(
                    gen_stats[0], gen_stats[1], gen_stats[2], gen_stats[3], gen_stats[4]
                )
            ]
        }
        
        # Save results
        results_file = os.path.join(self.evolutionary_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Evolutionary Algorithm completed in {runtime:.2f}s")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return results
    
    def run_population_based_training(self) -> Dict[str, Any]:
        """
        Run Population-Based Training using Ray Tune.
        
        Returns:
            Dictionary of results
        """
        logger.info(f"Starting Population-Based Training")
        start_time = time.time()
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        
        # Store results
        all_results = []
        
        # Define PBT objective function
        def pbt_objective(config: Dict[str, Any], checkpoint_dir: Optional[str] = None) -> None:
            # Extract trial info
            trial_id = tune.get_trial_id()
            seed = self.random_seed + int(trial_id.split("_")[-1])
            
            # Convert config to params (excluding seed)
            params = {k: v for k, v in config.items() if k != "seed"}
            
            # Create environment
            env_builder = EnvironmentBuilder()
            train_env = env_builder.build_env(
                self.env_id,
                render_mode=None
            )
            eval_env = env_builder.build_env(
                self.env_id,
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
                try:
                    agent.load(checkpoint_path)
                except:
                    logger.warning(f"Failed to load checkpoint from {checkpoint_path}")
            
            # Train agent for steps between tune reconfigurations
            steps_per_iter = 10000
            total_steps = 0
            max_steps = self.budget_per_trial
            
            # Metrics
            eval_freq = min(10000, steps_per_iter)  # steps between evaluations
            next_eval_at = eval_freq
            evaluation_rewards = []
            steps_to_threshold = None
            reward_threshold = 18.0  # for Pong, we consider solving at 18 points
            
            # Training loop
            while total_steps < max_steps:
                # Train for steps_per_iter or until max_steps
                steps_this_iter = min(steps_per_iter, max_steps - total_steps)
                if steps_this_iter <= 0:
                    break
                    
                # Reset environment for this iteration
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
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
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
        
        # Define hyperparameter space for Ray Tune
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
            "seed": self.random_seed
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
        population_size = min(self.n_trials, 10)  # Use a reasonable population size
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(pbt_objective),
                resources={"cpu": 1, "gpu": 0.25 if torch.cuda.is_available() else 0}
            ),
            run_config=tune.RunConfig(
                name="pbt_rainbow",
                local_dir=self.pbt_dir,
                stop={"total_steps": self.budget_per_trial}
            ),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=population_size
            ),
            param_space=param_space
        )
        
        try:
            results = tuner.fit()
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Get best trial
            best_trial = results.get_best_trial("mean_reward", "max", scope="all")
            best_params = {k: v for k, v in best_trial.config.items() if k != "seed"}
            best_score = best_trial.last_result["mean_reward"]
            
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
                    "detailed_metrics": detailed_metrics,
                    "checkpoint_path": os.path.abspath(os.path.join(trial.checkpoint.value, "checkpoint.pth")) 
                        if hasattr(trial, "checkpoint") and trial.checkpoint else None
                }
                all_results.append(result)
            
            # Prepare results
            output_results = {
                "method": "Population-Based Training",
                "best_params": best_params,
                "best_score": best_score,
                "runtime": runtime,
                "total_trials": len(all_results),
                "all_results": all_results,
                "configuration": {
                    "population_size": population_size,
                    "perturbation_interval": 10000
                }
            }
            
            # Save results
            results_file = os.path.join(self.pbt_dir, "results.json")
            with open(results_file, "w") as f:
                json.dump(output_results, f, indent=4)
            
            logger.info(f"Population-Based Training completed in {runtime:.2f}s")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Best params: {best_params}")
            
            return output_results
            
        finally:
            # Shutdown Ray
            ray.shutdown()
            
            # In case of errors, provide a default result
            if 'output_results' not in locals():
                output_results = {
                    "method": "Population-Based Training",
                    "best_params": {},
                    "best_score": float('-inf'),
                    "runtime": time.time() - start_time,
                    "total_trials": 0,
                    "all_results": [],
                    "error": "PBT execution failed"
                }
                
                results_file = os.path.join(self.pbt_dir, "results.json")
                with open(results_file, "w") as f:
                    json.dump(output_results, f, indent=4)
                
                return output_results
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all HPO methods and return their results.
        
        Returns:
            Dictionary mapping method names to their results
        """
        logger.info("Starting HPO comparison: running all methods")
        
        # Run Bayesian Optimization
        logger.info("Running Bayesian Optimization...")
        bayesian_results = self.run_bayesian_optimization()
        self.results["bayesian"] = bayesian_results
        
        # Run Evolutionary Algorithm
        logger.info("Running Evolutionary Algorithm...")
        evolutionary_results = self.run_evolutionary_algorithm()
        self.results["evolutionary"] = evolutionary_results
        
        # Run Population-Based Training
        logger.info("Running Population-Based Training...")
        pbt_results = self.run_population_based_training()
        self.results["pbt"] = pbt_results
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Path to the generated report
        """
        # Create analyzer for visualization
        analyzer = Analyzer(self.base_dir)
        analyzer.load_results()
        
        # Create report directory
        report_dir = os.path.join(self.base_dir, "comparison_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate plots
        figures_dir = os.path.join(report_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        analyzer.plot_learning_curves(save_path=os.path.join(figures_dir, "learning_curves.png"))
        analyzer.plot_method_comparison(save_path=os.path.join(figures_dir, "method_comparison.png"))
        analyzer.plot_hyperparameter_importance(save_path=os.path.join(figures_dir, "hyperparameter_importance.png"))
        analyzer.plot_reward_distributions(save_path=os.path.join(figures_dir, "reward_distributions.png"))
        
        # Generate comparison table as CSV and markdown
        comparison_df = analyzer.create_comparison_table()
        comparison_df.to_csv(os.path.join(report_dir, "comparison_table.csv"), index=False)
        
        # Get best hyperparameters for each method
        best_hyperparams = {}
        for method in self.results:
            if self.results[method]:
                best_hyperparams[method] = self.results[method].get("best_params", {})
        
        # Generate markdown report
        report_path = os.path.join(report_dir, "comparison_report.md")
        with open(report_path, "w") as f:
            f.write(f"# Hyperparameter Optimization Comparison for Rainbow DQN\n\n")
            f.write(f"## Environment: {self.env_id}\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration information
            f.write(f"## Experiment Configuration\n\n")
            f.write(f"- **Environment:** {self.env_id}\n")
            f.write(f"- **Number of trials:** {self.n_trials}\n")
            f.write(f"- **Number of seeds per trial:** {self.n_seeds}\n")
            f.write(f"- **Budget per trial:** {self.budget_per_trial} environment steps\n")
            f.write(f"- **Random seed:** {self.random_seed}\n\n")
            
            # Comparison table
            f.write(f"## Performance Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Learning curves
            f.write(f"## Learning Curves\n\n")
            f.write(f"![Learning Curves](figures/learning_curves.png)\n\n")
            
            # Method comparison bar chart
            f.write(f"## Method Comparison\n\n")
            f.write(f"![Method Comparison](figures/method_comparison.png)\n\n")
            
            # Parameter importance
            f.write(f"## Hyperparameter Importance\n\n")
            f.write(f"![Hyperparameter Importance](figures/hyperparameter_importance.png)\n\n")
            
            # Reward distributions
            f.write(f"## Reward Distributions\n\n")
            f.write(f"![Reward Distributions](figures/reward_distributions.png)\n\n")
            
            # Best hyperparameters for each method
            f.write(f"## Best Hyperparameters\n\n")
            for method, params in best_hyperparams.items():
                f.write(f"### {method.capitalize()}\n\n")
                f.write("```json\n")
                f.write(json.dumps(params, indent=2))
                f.write("\n```\n\n")
            
            # Comprehensive analysis
            f.write(f"## Analysis Summary\n\n")
            
            # Performance
            f.write(f"### Performance Metrics\n\n")
            
            # Extract best scores
            best_scores = {}
            for method in self.results:
                if self.results[method]:
                    best_scores[method] = self.results[method].get("best_score", float('-inf'))
            
            # Find best method
            best_method = max(best_scores.items(), key=lambda x: x[1])[0] if best_scores else None
            
            if best_method:
                f.write(f"The best performing method was **{best_method.capitalize()}** ")
                f.write(f"with a mean reward of **{best_scores[best_method]:.2f}**.\n\n")
            
            # Runtime comparison
            runtimes = {}
            for method in self.results:
                if self.results[method]:
                    runtimes[method] = self.results[method].get("runtime", float('inf'))
            
            fastest_method = min(runtimes.items(), key=lambda x: x[1])[0] if runtimes else None
            
            if fastest_method:
                f.write(f"The fastest method was **{fastest_method.capitalize()}** ")
                f.write(f"with a runtime of **{runtimes[fastest_method]:.2f}** seconds.\n\n")
            
            # Method-specific insights
            f.write(f"### Method-Specific Insights\n\n")
            
            # Bayesian Optimization
            if "bayesian" in self.results and self.results["bayesian"]:
                f.write(f"#### Bayesian Optimization\n\n")
                
                # Parameter importance
                param_importances = self.results["bayesian"].get("param_importances", {})
                if param_importances:
                    f.write(f"Top important hyperparameters:\n\n")
                    sorted_params = sorted(param_importances.items(), key=lambda x: x[1], reverse=True)[:3]
                    for param, importance in sorted_params:
                        f.write(f"- {param}: {importance:.4f}\n")
                    f.write("\n")
            
            # Evolutionary Algorithm
            if "evolutionary" in self.results and self.results["evolutionary"]:
                f.write(f"#### Evolutionary Algorithm\n\n")
                
                # Generation statistics
                gen_stats = self.results["evolutionary"].get("gen_stats", [])
                if gen_stats:
                    first_gen = gen_stats[0]
                    last_gen = gen_stats[-1]
                    
                    avg_improvement = last_gen.get("avg", 0) - first_gen.get("avg", 0)
                    max_improvement = last_gen.get("max", 0) - first_gen.get("max", 0)
                    
                    f.write(f"Over {len(gen_stats)} generations, the average fitness improved by ")
                    f.write(f"**{avg_improvement:.2f}** and the maximum fitness improved by ")
                    f.write(f"**{max_improvement:.2f}**.\n\n")
            
            # Population-Based Training
            if "pbt" in self.results and self.results["pbt"]:
                f.write(f"#### Population-Based Training\n\n")
                
                # Population size
                pop_size = self.results["pbt"].get("configuration", {}).get("population_size", 0)
                f.write(f"PBT used a population of {pop_size} agents, each exploring different ")
                f.write(f"regions of the hyperparameter space while dynamically adapting ")
                f.write(f"through the training process.\n\n")
            
            # Conclusion
            f.write(f"## Conclusion\n\n")
            f.write(f"This experiment compared three hyperparameter optimization strategies ")
            f.write(f"for Rainbow DQN on the {self.env_id} environment. ")
            
            if best_method:
                f.write(f"Overall, **{best_method.capitalize()}** achieved the best performance, ")
                f.write(f"but each method has its own strengths and trade-offs in terms of ")
                f.write(f"computational efficiency, exploration capability, and robustness.")
            
            f.write("\n\n---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Comparison report generated at {report_path}")
        return report_path


def main():
    """Main function for running the HPO comparison."""
    parser = argparse.ArgumentParser(description="Compare HPO methods for Rainbow DQN")
    
    parser.add_argument(
        "--env", 
        type=str, 
        default="PongNoFrameskip-v4",
        help="Gym environment ID"
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
        help="Number of seeds per trial for statistical reliability"
    )
    
    parser.add_argument(
        "--budget", 
        type=int, 
        default=500000,
        help="Training budget per trial in environment steps"
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
        default="hpo_comparison_results",
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--bayesian", 
        action="store_true",
        help="Run only Bayesian Optimization"
    )
    
    parser.add_argument(
        "--evolutionary", 
        action="store_true",
        help="Run only Evolutionary Algorithm"
    )
    
    parser.add_argument(
        "--pbt", 
        action="store_true",
        help="Run only Population-Based Training"
    )
    
    args = parser.parse_args()
    
    # Create HPO comparison
    comparison = HPOComparison(
        env_id=args.env,
        n_trials=args.trials,
        n_seeds=args.seeds,
        budget_per_trial=args.budget,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Run specific methods or all if none specified
    if args.bayesian:
        comparison.results["bayesian"] = comparison.run_bayesian_optimization()
    elif args.evolutionary:
        comparison.results["evolutionary"] = comparison.run_evolutionary_algorithm()
    elif args.pbt:
        comparison.results["pbt"] = comparison.run_population_based_training()
    else:
        # Run all methods
        comparison.run_all()
    
    # Generate comparison report
    comparison.generate_comparison_report()


if __name__ == "__main__":
    main()