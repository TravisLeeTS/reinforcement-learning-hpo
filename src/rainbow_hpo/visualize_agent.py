"""
Visualization script for Rainbow DQN trained models.
This module lets you watch how trained models play in their environments.
"""

import os
import argparse
import logging
import gymnasium as gym
import numpy as np
import torch
import time
import glob
import json
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

from agent_builder import RainbowDQNAgent, AgentBuilder
from env_builder import EnvironmentBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def list_available_models(models_dir: str = "models"):
    """
    List all available trained models in the models directory.
    
    Args:
        models_dir: Directory containing trained models
    """
    # Look for best_model.zip files recursively
    model_paths = glob.glob(os.path.join(models_dir, "**/best_model.zip"), recursive=True)
    model_paths += glob.glob(os.path.join(models_dir, "**/final_model.zip"), recursive=True)
    
    if not model_paths:
        logger.warning(f"No models found in {models_dir} directory.")
        return []
    
    # Sort by directory structure
    model_paths.sort()
    
    # Print available models
    logger.info(f"Found {len(model_paths)} trained models:")
    for i, path in enumerate(model_paths):
        logger.info(f"[{i}] {path}")
    
    return model_paths


def visualize_agent(
    model_path: str,
    env_id: Optional[str] = None,
    num_episodes: int = 5,
    max_steps: int = 1000,
    render_mode: str = "human",
    delay: float = 0.02
):
    """
    Visualize a trained agent playing in its environment.
    
    Args:
        model_path: Path to the trained model file
        env_id: Environment ID (if None, try to infer from model path or config)
        num_episodes: Number of episodes to play
        max_steps: Maximum steps per episode
        render_mode: Rendering mode ('human' or 'rgb_array')
        delay: Delay between steps in seconds (to slow down visualization)
    """
    # Try to infer environment ID if not provided
    if env_id is None:
        # Check for config.json in parent directories
        model_dir = os.path.dirname(model_path)
        config_paths = [
            os.path.join(model_dir, "config.json"),
            os.path.join(os.path.dirname(model_dir), "config.json"),
            os.path.join(os.path.dirname(os.path.dirname(model_dir)), "config.json")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'env_id' in config:
                        env_id = config['env_id']
                        logger.info(f"Inferred environment ID from config: {env_id}")
                        break
        
        # If still not found, try to guess from directory name
        if env_id is None:
            # Check for common environment names in the model path
            common_envs = ["Pong", "Breakout", "SpaceInvaders", "CartPole", "Pendulum", "MountainCar", "Acrobot"]
            for env_name in common_envs:
                if env_name.lower() in model_path.lower():
                    if "Pong" in env_name:
                        env_id = "PongNoFrameskip-v4"
                    elif "Breakout" in env_name:
                        env_id = "BreakoutNoFrameskip-v4"
                    elif "SpaceInvaders" in env_name:
                        env_id = "SpaceInvadersNoFrameskip-v4"
                    else:
                        env_id = f"{env_name}-v1"
                    logger.info(f"Guessed environment ID from model path: {env_id}")
                    break
    
    if env_id is None:
        logger.error("Could not determine environment ID. Please specify with --env argument.")
        return
    
    # Create environment
    env_builder = EnvironmentBuilder()
    env = env_builder.build_env(
        env_id,
        render_mode=render_mode,
        max_episode_steps=max_steps
    )
    
    # Create agent builder
    agent_builder = AgentBuilder()
    
    # Create agent
    agent = agent_builder.build_agent(
        env.observation_space,
        env.action_space,
        {}, # Empty hyperparameters dict, will be overridden by loaded model
        seed=42
    )
    
    # Load trained model
    try:
        agent.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        env.close()
        return
    
    # Play episodes
    total_reward = 0
    episode_rewards = []
    
    logger.info(f"Starting visualization of {num_episodes} episodes in {env_id}")
    logger.info(f"Press Ctrl+C to stop...")
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            step = 0
            done = False
            
            logger.info(f"Episode {episode+1}/{num_episodes}")
            
            while not done and step < max_steps:
                # Select action using trained policy
                action = agent.select_action(state, eval_mode=True)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                step += 1
                
                # Delay to make visualization viewable
                if delay > 0:
                    time.sleep(delay)
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            
            logger.info(f"Episode {episode+1} finished with reward: {episode_reward:.2f}, steps: {step}")
            
            # Short pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    
    finally:
        # Close environment
        env.close()
    
    # Print statistics
    avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0
    logger.info(f"Visualization completed.")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Episode rewards: {episode_rewards}")
    
    return episode_rewards


def main():
    """Main function for the visualization script."""
    parser = argparse.ArgumentParser(description="Rainbow DQN Model Visualization")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--env", type=str, help="Environment ID (if not specified, will try to infer)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps (seconds)")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--model-index", type=int, help="Index of the model to use (from --list)")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list or (args.model is None and args.model_index is None):
        models = list_available_models(args.models_dir)
        if not models:
            return
        
        # If model_index is provided, use that model
        if args.model_index is not None:
            if 0 <= args.model_index < len(models):
                args.model = models[args.model_index]
            else:
                logger.error(f"Invalid model index: {args.model_index}. Must be between 0 and {len(models)-1}")
                return
        else:
            # Ask user to pick a model
            try:
                idx = int(input(f"Enter model number (0-{len(models)-1}): "))
                if 0 <= idx < len(models):
                    args.model = models[idx]
                else:
                    logger.error(f"Invalid model index: {idx}. Must be between 0 and {len(models)-1}")
                    return
            except ValueError:
                logger.error("Invalid input. Please enter a number.")
                return
            except KeyboardInterrupt:
                logger.info("Visualization canceled by user")
                return
    
    # Run visualization
    if args.model:
        visualize_agent(
            model_path=args.model,
            env_id=args.env,
            num_episodes=args.episodes,
            max_steps=args.steps,
            delay=args.delay
        )
    else:
        logger.error("No model specified. Use --model or --list options.")


if __name__ == "__main__":
    main()