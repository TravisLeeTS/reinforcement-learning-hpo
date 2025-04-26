"""
Main script for running Rainbow DQN experiments with hyperparameter optimization.
This module serves as the central entry point for running various Rainbow DQN experiments.
"""

import gymnasium as gym
import numpy as np
import argparse
import os
import logging
import json
from pathlib import Path

from utils.common import set_seed, setup_logging, create_directory_if_not_exists
from hpo_engine import HyperparameterOptimizer
from training import train_with_optimal_params, optimize_hyperparameters

# Configure logging
setup_logging(log_dir="logs")
logger = logging.getLogger(__name__)


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