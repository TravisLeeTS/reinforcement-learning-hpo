"""
Main script for running Rainbow DQN experiments with hyperparameter optimization.
This module serves as the central entry point for running various Rainbow DQN experiments.
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
import random

from agent_builder import RainbowDQNAgent, AgentBuilder, AugmentationConfig
from hpo_engine import HyperparameterOptimizer
from env_builder import EnvironmentBuilder
from analyzer import Analyzer

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
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


def optimize_hyperparameters(n_trials: int = 10, seed: int = 42, env_id: str = "Pendulum-v1", output_dir: str = "models"):
    """
    Run hyperparameter optimization for Rainbow DQN.
    
    Args:
        n_trials: Number of optimization trials
        seed: Random seed
        env_id: Environment ID
        output_dir: Output directory
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Define hyperparameter search space based on environment type
    is_atari = env_id.startswith("ALE/") or "NoFrameskip" in env_id
    
    if is_atari:
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
            "v_max": ("uniform", 5.0, 20.0),
            "max_episodes": ("int", 200, 500)
        }
    else:
        param_space = {
            "learning_rate": ("log_uniform", 1e-5, 1e-3),
            "gamma": ("uniform", 0.95, 0.99),
            "batch_size": ("categorical", [32, 64, 128]),
            "target_update_freq": ("int", 500, 2000),
            "n_steps": ("int", 1, 5),
            "epsilon_start": ("uniform", 0.5, 1.0),
            "epsilon_end": ("uniform", 0.01, 0.1),
            "epsilon_decay_steps": ("int", 10000, 100000),
            "n_atoms": ("categorical", [21, 51, 101]),
            "v_min": ("uniform", -20.0, -5.0),
            "v_max": ("uniform", 5.0, 20.0),
            "max_episodes": ("int", 300, 800)
        }
    
    # Create objective function with fixed env_id and output_dir
    objective = lambda params, trial_id: objective_function(params, trial_id, env_id, output_dir)
    
    # Initialize optimizer
    hpo = HyperparameterOptimizer(
        param_space=param_space,
        objective_function=objective,
        n_trials=n_trials,
        seed=seed,
        strategy="tpe"  # Tree-structured Parzen Estimator approach
    )
    
    # Run optimization
    best_params, best_value = hpo.optimize()
    
    logger.info(f"Optimization completed.")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best evaluation reward: {best_value:.2f}")
    
    # Save best hyperparameters
    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    # Return best parameters and score for potential further use
    return best_params, best_value


def train_with_optimal_params(env_id: str = "Pendulum-v1", render: bool = True, output_dir: str = "models"):
    """
    Train agent using the best found hyperparameters.
    
    Args:
        env_id: Environment ID
        render: Whether to render the environment
        output_dir: Output directory
    """
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


def list_atari_games():
    """List available Atari games in Gymnasium."""
    import gymnasium as gym
    from gymnasium.envs.registration import registry
    
    atari_envs = [env_id for env_id in registry.keys() if 'ALE/' in env_id and not env_id.endswith('-ram')]
    
    print("\nAvailable Atari Environments:")
    print("============================")
    for env_id in sorted(atari_envs):
        print(f"  - {env_id}")
    print("\nUsage example: --env ALE/Pong-v5")


def main():
    """Main function to run Rainbow DQN experiments."""
    parser = argparse.ArgumentParser(description="Rainbow DQN with Hyperparameter Optimization")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--train", action="store_true", help="Train with optimal hyperparameters")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of HPO trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--render", action="store_true", help="Render environment when training")
    parser.add_argument("--list-games", action="store_true", help="List available Atari games and exit")
    
    args = parser.parse_args()
    
    if args.list_games:
        list_atari_games()
        return
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    if args.optimize:
        optimize_hyperparameters(
            n_trials=args.n_trials, 
            seed=args.seed, 
            env_id=args.env,
            output_dir=args.output_dir
        )
    
    if args.train:
        train_with_optimal_params(
            env_id=args.env,
            render=args.render,
            output_dir=args.output_dir
        )
    
    if not args.optimize and not args.train and not args.list_games:
        logger.warning("No action specified. Use --optimize or --train")
        parser.print_help()


if __name__ == "__main__":
    main()