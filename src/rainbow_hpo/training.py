"""
Training module for Rainbow DQN agent.
Contains functions for training, evaluating, and managing training processes.
"""

import gymnasium as gym
import numpy as np
import torch
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

from agent_builder import RainbowDQNAgent
from env_builder import EnvironmentBuilder

logger = logging.getLogger(__name__)


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
    std_reward = np.std(episode_rewards)
    logger.info(f"Evaluation over {n_episodes} episodes: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward


def train_agent(
    agent: RainbowDQNAgent,
    env: gym.Env, 
    max_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    eval_freq: int = 50,
    save_freq: int = 100,
    render: bool = False,
    trial_id: Union[int, str] = 0,
    output_dir: str = "models",
    eval_env: Optional[gym.Env] = None,
    early_stopping_threshold: Optional[float] = None,
    patience: int = 3
) -> Tuple[List[float], List[float], float]:
    """
    Train a Rainbow DQN agent.
    
    Args:
        agent: Rainbow DQN agent
        env: Training Gym environment
        max_episodes: Maximum number of episodes
        max_steps_per_episode: Maximum steps per episode
        eval_freq: Evaluation frequency (in episodes)
        save_freq: Model saving frequency (in episodes)
        render: Whether to render the environment
        trial_id: Trial ID
        output_dir: Output directory
        eval_env: Optional separate environment for evaluation
        early_stopping_threshold: Reward threshold for early stopping (None to disable)
        patience: Number of consecutive evaluations without improvement before stopping
        
    Returns:
        Tuple of (episode_rewards, training_losses, best_eval_reward)
    """
    episode_rewards = []
    training_losses = []
    best_eval_reward = float('-inf')
    start_time = time.time()
    
    trial_str = str(trial_id)
    logger.info(f"Starting training for trial {trial_str}")
    
    # Create trial directory
    trial_dir = os.path.join(output_dir, f"trial_{trial_str}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Progress tracking
    progress_bar = tqdm(total=max_episodes, desc=f"Training (Trial {trial_str})")
    
    # Use separate evaluation environment if provided
    if eval_env is None:
        eval_env = env
    
    # Early stopping variables
    no_improvement_count = 0
    threshold_reached = False
    
    # Training loop
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
                
            if render and (episode % 100 == 0):  # Only render occasionally for long training runs
                env.render()
        
        # Append episode reward
        episode_rewards.append(episode_reward)
        
        # Update progress bar
        progress_bar.update(1)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_loss = episode_loss / max(1, n_steps)
            progress_bar.set_description(
                f"Trial {trial_str} | Ep {episode}/{max_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg(10): {avg_reward:.2f} | "
                f"Eps: {agent.epsilon:.3f}"
            )
            
            # Log more detailed info
            logger.info(
                f"Episode {episode}/{max_episodes} | "
                f"Steps: {n_steps} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward (10 ep): {avg_reward:.2f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )
        
        # Periodic evaluation
        if episode > 0 and episode % eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env)
            
            # Check for improvement
            if eval_reward > best_eval_reward:
                # Reset counter on improvement
                no_improvement_count = 0
                best_eval_reward = eval_reward
                agent.save(os.path.join(trial_dir, "best_model.zip"))
                logger.info(f"New best model saved with eval reward: {best_eval_reward:.2f}")
            else:
                # Increment counter on no improvement
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} evaluations. Best: {best_eval_reward:.2f}, Current: {eval_reward:.2f}")
            
            # Early stopping based on threshold
            if early_stopping_threshold is not None and eval_reward >= early_stopping_threshold:
                logger.info(f"Early stopping threshold {early_stopping_threshold} reached with reward {eval_reward:.2f}")
                threshold_reached = True
                break
            
            # Early stopping based on patience
            if patience > 0 and no_improvement_count >= patience:
                logger.info(f"Early stopping due to no improvement for {patience} evaluations")
                break
        
        # Periodic model saving
        if episode > 0 and episode % save_freq == 0:
            agent.save(os.path.join(trial_dir, f"checkpoint_ep{episode}.zip"))
    
    # Close progress bar
    progress_bar.close()
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Final model save
    agent.save(os.path.join(trial_dir, "final_model.zip"))
    
    # Final evaluation
    final_eval_reward = evaluate_agent(agent, eval_env, n_episodes=20)
    if final_eval_reward > best_eval_reward:
        best_eval_reward = final_eval_reward
        agent.save(os.path.join(trial_dir, "best_model.zip"))
    
    # Save early stopping information
    early_stopping_info = {
        "threshold_reached": threshold_reached,
        "early_stopping_threshold": early_stopping_threshold,
        "patience": patience,
        "best_eval_reward": best_eval_reward,
        "final_eval_reward": final_eval_reward,
        "training_time_seconds": training_time,
        "completed_episodes": episode + 1
    }
    
    with open(os.path.join(trial_dir, "early_stopping_info.json"), "w") as f:
        json.dump(early_stopping_info, f, indent=4)
    
    return episode_rewards, training_losses, best_eval_reward


def save_training_metrics(rewards: List[float], losses: List[float], trial_id: int, output_dir: str = "models") -> None:
    """
    Save training metrics and generate plots.
    
    Args:
        rewards: Episode rewards
        losses: Training losses
        trial_id: Trial ID
        output_dir: Output directory
    """
    # Create directory for trial metrics
    metrics_dir = os.path.join(output_dir, f"trial_{trial_id}", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save raw data
    rewards_df = pd.DataFrame({"episode": range(len(rewards)), "reward": rewards})
    rewards_df.to_csv(os.path.join(metrics_dir, "rewards.csv"), index=False)
    
    if losses:
        losses_df = pd.DataFrame({"step": range(len(losses)), "loss": losses})
        losses_df.to_csv(os.path.join(metrics_dir, "losses.csv"), index=False)
    
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
    plt.savefig(os.path.join(metrics_dir, "training_metrics.png"))
    plt.close()
    
    logger.info(f"Training metrics saved to {metrics_dir}")


def save_hyperparameters(hyperparams: Dict[str, Any], trial_id: int, output_dir: str = "models") -> None:
    """
    Save hyperparameters to file.
    
    Args:
        hyperparams: Hyperparameters dictionary
        trial_id: Trial ID
        output_dir: Output directory
    """
    trial_dir = os.path.join(output_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    path = os.path.join(trial_dir, "hyperparameters.json")
    
    with open(path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    logger.info(f"Hyperparameters saved to {path}")


def objective_function(hyperparams: Dict[str, Any], trial_id: int, env_id: str = "Pendulum-v1", output_dir: str = "models") -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        hyperparams: Hyperparameters dictionary
        trial_id: Trial ID
        env_id: Environment ID
        output_dir: Output directory
        
    Returns:
        Mean evaluation reward (to be maximized)
    """
    from agent_builder import AgentBuilder
    
    # Create directories
    trial_dir = os.path.join(output_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Log hyperparameters
    logger.info(f"Trial {trial_id} hyperparameters: {hyperparams}")
    save_hyperparameters(hyperparams, trial_id, output_dir)
    
    # Create environment
    env_builder = EnvironmentBuilder()
    train_env = env_builder.build_env(env_id, render_mode=None)
    eval_env = env_builder.build_env(env_id, render_mode=None)
    
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
    
    # Set early stopping threshold based on environment
    early_stopping_threshold = None
    patience = 3  # Default patience
    
    # Environment-specific early stopping settings
    if "Pong" in env_id:
        early_stopping_threshold = 18.0  # Good performance for Pong
        patience = 5
    elif "Breakout" in env_id:
        early_stopping_threshold = 30.0
        patience = 5
    elif "Pendulum" in env_id:
        early_stopping_threshold = -200.0  # Pendulum is negative reward, higher is better
        patience = 3
    elif "CartPole" in env_id:
        early_stopping_threshold = 475.0  # Max is 500, so 475 is very good
        patience = 3
    elif "MountainCar" in env_id:
        early_stopping_threshold = -110.0  # Higher is better
        patience = 3
    elif "Acrobot" in env_id:
        early_stopping_threshold = -100.0  # Higher is better
        patience = 3
    
    # Train agent
    max_episodes = hyperparams.get("max_episodes", 500)
    episode_rewards, training_losses, best_eval_reward = train_agent(
        agent=agent,
        env=train_env,
        eval_env=eval_env,
        max_episodes=max_episodes,
        max_steps_per_episode=1000,
        eval_freq=50,
        save_freq=100,
        render=False,
        trial_id=trial_id,
        output_dir=output_dir,
        early_stopping_threshold=early_stopping_threshold,
        patience=patience
    )
    
    # Save training metrics
    save_training_metrics(episode_rewards, training_losses, trial_id, output_dir)
    
    # Final evaluation
    final_eval_reward = evaluate_agent(agent, eval_env, n_episodes=20)
    logger.info(f"Trial {trial_id} final evaluation reward: {final_eval_reward:.2f}")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    # Return the best of the evaluation rewards
    return max(best_eval_reward, final_eval_reward)


def train_with_optimal_params(env_id: str = "Pendulum-v1", render: bool = True, output_dir: str = "models"):
    """
    Train agent using the best found hyperparameters.
    
    Args:
        env_id: Environment ID
        render: Whether to render the environment
        output_dir: Output directory
    """
    from agent_builder import AgentBuilder
    from utils.common import set_seed
    
    # Load best hyperparameters
    hyperparams_path = os.path.join(output_dir, "best_hyperparameters.json")
    
    if not os.path.exists(hyperparams_path):
        logger.error(f"Best hyperparameters file not found at {hyperparams_path}")
        return
    
    with open(hyperparams_path, "r") as f:
        best_params = json.load(f)
    
    logger.info(f"Training with optimal hyperparameters: {best_params}")
    
    # Create environment
    env_builder = EnvironmentBuilder()
    render_mode = "human" if render else None
    env = env_builder.build_env(env_id, render_mode=render_mode)
    eval_env = env_builder.build_env(env_id, render_mode=None)
    
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    
    # Build agent
    agent_builder = AgentBuilder()
    agent = agent_builder.build_agent(
        env.observation_space,
        env.action_space,
        best_params,
        seed=seed
    )
    
    # Train agent
    max_episodes = best_params.get("max_episodes", 1000)
    train_agent(
        agent=agent,
        env=env,
        eval_env=eval_env,
        max_episodes=max_episodes,
        max_steps_per_episode=1000,
        eval_freq=50,
        save_freq=100,
        render=render,
        trial_id="optimal",
        output_dir=output_dir
    )
    
    env.close()
    eval_env.close()


def optimize_hyperparameters(n_trials: int = 10, seed: int = 42, env_id: str = "Pendulum-v1", output_dir: str = "models") -> None:
    """
    Run hyperparameter optimization to find the best hyperparameters.
    
    Args:
        n_trials: Number of trials to run
        seed: Random seed for reproducibility
        env_id: Environment ID
        output_dir: Output directory
    """
    from hpo_engine import HyperparameterOptimizer
    from utils.common import set_seed
    
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    set_seed(seed)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        objective_fn=objective_function,
        env_id=env_id,
        output_dir=output_dir
    )
    
    # Run optimization
    best_params, best_value = optimizer.optimize(n_trials=n_trials)
    
    # Save best hyperparameters
    best_params_path = os.path.join(output_dir, "best_hyperparameters.json")
    best_result_dict = {
        "best_params": best_params,
        "best_value": best_value,
        "env_id": env_id,
        "n_trials": n_trials,
        "seed": seed
    }
    
    with open(best_params_path, "w") as f:
        json.dump(best_result_dict, f, indent=4)
    
    logger.info(f"Best hyperparameters saved to {best_params_path}")
    logger.info(f"Best value: {best_value}")
    logger.info(f"Best parameters: {best_params}")
    
    return best_params, best_value