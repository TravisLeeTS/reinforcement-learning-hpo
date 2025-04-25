"""
Main script for running Rainbow DQN with advanced features from DI-engine and Pearl.
This module integrates the latest improvements from state-of-the-art RL libraries.
"""

import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from agent_builder import RainbowDQNAgent, AgentBuilder, AugmentationConfig
from hpo_engine import HyperparameterOptimizer
from env_builder import EnvironmentBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directory_if_not_exists(path: str) -> None:
    """
    Create directory if it does not already exist.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Created directory: {path}")


def save_hyperparameters(hyperparams: Dict[str, Any], trial_id: int) -> None:
    """
    Save hyperparameters to file.
    
    Args:
        hyperparams: Hyperparameters dictionary
        trial_id: Trial ID
    """
    path = f"models/trial_{trial_id}/hyperparameters.json"
    with open(path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    logger.info(f"Hyperparameters saved to {path}")


def save_training_metrics(rewards: List[float], losses: List[float], trial_id: int) -> None:
    """
    Save training metrics and generate plots.
    
    Args:
        rewards: Episode rewards
        losses: Training losses
        trial_id: Trial ID
    """
    # Create directory for trial metrics
    metrics_dir = f"models/trial_{trial_id}/metrics"
    create_directory_if_not_exists(metrics_dir)
    
    # Save raw data
    rewards_df = pd.DataFrame({"episode": range(len(rewards)), "reward": rewards})
    rewards_df.to_csv(f"{metrics_dir}/rewards.csv", index=False)
    
    if losses:
        losses_df = pd.DataFrame({"step": range(len(losses)), "loss": losses})
        losses_df.to_csv(f"{metrics_dir}/losses.csv", index=False)
    
    # Generate plots
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    sns.lineplot(data=rewards_df, x="episode", y="reward")
    plt.title(f"Episode Rewards (Trial {trial_id})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot losses if available
    if losses:
        plt.subplot(1, 2, 2)
        # Smoothen losses for better visualization
        window_size = min(100, max(1, len(losses) // 100))
        losses_df["smoothed_loss"] = losses_df["loss"].rolling(window=window_size).mean()
        sns.lineplot(data=losses_df, x="step", y="smoothed_loss")
        plt.title(f"Training Losses (Trial {trial_id})")
        plt.xlabel("Training Step")
        plt.ylabel("Loss (smoothed)")
    
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/training_metrics.png")
    plt.close()
    
    logger.info(f"Training metrics saved to {metrics_dir}")


def evaluate_agent(agent: RainbowDQNAgent, env: gym.Env, n_episodes: int = 10) -> float:
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
    logger.info(f"Evaluation over {n_episodes} episodes: Mean reward = {mean_reward:.2f}")
    
    return mean_reward


def train_agent(
    agent: RainbowDQNAgent,
    env: gym.Env, 
    max_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    eval_freq: int = 50,
    save_freq: int = 100,
    render: bool = False,
    trial_id: int = 0
) -> Tuple[List[float], List[float]]:
    """
    Train a Rainbow DQN agent.
    
    Args:
        agent: Rainbow DQN agent
        env: Gym environment
        max_episodes: Maximum number of episodes
        max_steps_per_episode: Maximum steps per episode
        eval_freq: Evaluation frequency (in episodes)
        save_freq: Model saving frequency (in episodes)
        render: Whether to render the environment
        trial_id: Trial ID
        
    Returns:
        Tuple of (episode_rewards, training_losses)
    """
    episode_rewards = []
    training_losses = []
    best_eval_reward = float('-inf')
    start_time = time.time()
    
    logger.info(f"Starting training for trial {trial_id}")
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        n_steps = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Optimize (train)
            loss = agent.optimize()
            if loss != 0:  # Only track non-zero losses (when optimization actually happens)
                training_losses.append(loss)
                episode_loss += loss
            
            episode_reward += reward
            state = next_state
            n_steps += 1
            
            if done:
                break
                
            if render and episode % 100 == 0:  # Only render occasionally for long training runs
                env.render()
        
        # Append episode reward
        episode_rewards.append(episode_reward)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = episode_loss / n_steps if n_steps > 0 else 0
            logger.info(f"Episode {episode}/{max_episodes} | "
                       f"Steps: {n_steps} | "
                       f"Reward: {episode_reward:.2f} | "
                       f"Avg Reward (10 ep): {avg_reward:.2f} | "
                       f"Avg Loss: {avg_loss:.4f} | "
                       f"Epsilon: {agent.epsilon:.4f}")
        
        # Periodic evaluation
        if episode > 0 and episode % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/trial_{trial_id}/best_model.zip")
                logger.info(f"New best model saved with eval reward: {best_eval_reward:.2f}")
        
        # Periodic model saving
        if episode > 0 and episode % save_freq == 0:
            agent.save(f"models/trial_{trial_id}/checkpoint_ep{episode}.zip")
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Final model save
    agent.save(f"models/trial_{trial_id}/final_model.zip")
    
    return episode_rewards, training_losses


def objective_function(hyperparams: Dict[str, Any], trial_id: int) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        hyperparams: Hyperparameters dictionary
        trial_id: Trial ID
        
    Returns:
        Mean evaluation reward (to be maximized)
    """
    # Create directories
    trial_dir = f"models/trial_{trial_id}"
    create_directory_if_not_exists(trial_dir)
    
    # Log hyperparameters
    logger.info(f"Trial {trial_id} hyperparameters: {hyperparams}")
    save_hyperparameters(hyperparams, trial_id)
    
    # Create environment
    env_builder = EnvironmentBuilder()
    train_env = env_builder.build_env("Pendulum-v1", render_mode=None)
    eval_env = env_builder.build_env("Pendulum-v1", render_mode=None)
    
    # Set seeds
    seed = 42 + trial_id
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed)
    
    # Build agent
    agent_builder = AgentBuilder()
    agent = agent_builder.build_agent(
        train_env.observation_space,
        train_env.action_space,
        hyperparams,
        seed=seed
    )
    
    # Train agent
    max_episodes = hyperparams.get("max_episodes", 500)
    episode_rewards, training_losses = train_agent(
        agent=agent,
        env=train_env,
        max_episodes=max_episodes,
        max_steps_per_episode=1000,
        eval_freq=50,
        save_freq=100,
        render=False,
        trial_id=trial_id
    )
    
    # Save training metrics
    save_training_metrics(episode_rewards, training_losses, trial_id)
    
    # Final evaluation
    final_eval_reward = evaluate_agent(agent, eval_env, n_episodes=20)
    logger.info(f"Trial {trial_id} final evaluation reward: {final_eval_reward:.2f}")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    return final_eval_reward


def optimize_hyperparameters(n_trials: int = 10, seed: int = 42):
    """
    Run hyperparameter optimization for Rainbow DQN.
    
    Args:
        n_trials: Number of optimization trials
        seed: Random seed
    """
    # Create directories
    create_directory_if_not_exists("models")
    create_directory_if_not_exists("logs")
    
    # Define hyperparameter search space
    param_space = {
        "learning_rate": ("log_uniform", 1e-5, 1e-3),
        "gamma": ("uniform", 0.95, 0.99),
        "batch_size": ("categorical", [32, 64, 128]),
        "target_update_freq": ("int", 500, 2000),
        "n_steps": ("int", 1, 5),
        "n_atoms": ("categorical", [21, 51, 101]),
        "v_min": ("uniform", -20.0, -5.0),
        "v_max": ("uniform", 5.0, 20.0)
    }
    
    # Initialize optimizer
    hpo = HyperparameterOptimizer(
        param_space=param_space,
        objective_function=objective_function,
        n_trials=n_trials,
        seed=seed
    )
    
    # Run optimization
    best_params, best_value = hpo.optimize()
    
    logger.info(f"Optimization completed.")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best evaluation reward: {best_value:.2f}")
    
    # Save best hyperparameters
    with open("models/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)


def train_with_optimal_params():
    """Train agent using the best found hyperparameters."""
    # Load best hyperparameters
    with open("models/best_hyperparameters.json", "r") as f:
        best_params = json.load(f)
    
    logger.info(f"Training with optimal hyperparameters: {best_params}")
    
    # Create environment
    env_builder = EnvironmentBuilder()
    env = env_builder.build_env("Pendulum-v1", render_mode="human")
    
    # Build agent
    agent_builder = AgentBuilder()
    agent = agent_builder.build_agent(
        env.observation_space,
        env.action_space,
        best_params,
        seed=42
    )
    
    # Train agent
    train_agent(
        agent=agent,
        env=env,
        max_episodes=1000,
        max_steps_per_episode=1000,
        eval_freq=50,
        save_freq=100,
        render=True,
        trial_id="optimal"
    )
    
    env.close()


def main():
    """Main function to run Rainbow DQN experiments."""
    parser = argparse.ArgumentParser(description="Rainbow DQN with Hyperparameter Optimization")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--train", action="store_true", help="Train with optimal hyperparameters")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of HPO trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_hyperparameters(n_trials=args.n_trials, seed=args.seed)
    
    if args.train:
        train_with_optimal_params()
    
    if not args.optimize and not args.train:
        logger.warning("No action specified. Use --optimize or --train")
        parser.print_help()


if __name__ == "__main__":
    main()