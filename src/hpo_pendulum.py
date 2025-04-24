import os
import gymnasium as gym
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Callable
import time
import subprocess
import sys

# Set up logging directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Define a pause callback
class PauseCallback(BaseCallback):
    """
    Custom callback that pauses training when a reward threshold is reached.
    """
    def __init__(self, reward_threshold: float = 475.0, check_freq: int = 1000, verbose: int = 0):
        super(PauseCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.current_mean_reward = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate the current policy
            eval_env = gym.make('CartPole-v1')
            mean_reward, _ = evaluate_policy(self.model, eval_env, n_eval_episodes=5, deterministic=True)
            eval_env.close()
            self.current_mean_reward = mean_reward
            
            if mean_reward >= self.reward_threshold:
                print(f"\nReward threshold {self.reward_threshold} reached! Mean reward: {mean_reward:.2f}")
                print("Pausing training for 5 seconds...")
                time.sleep(5)
                print("Resuming training...")

        return True  # Always continue training

def create_env():
    """Create and wrap the environment."""
    # Using CartPole instead of Pendulum to avoid Box2D dependency issues
    env = gym.make('CartPole-v1')
    env = Monitor(env, log_dir)
    return env

def optimize_ppo(trial: optuna.Trial) -> float:
    """Optimize PPO hyperparameters for CartPole-v1."""
    # Hyperparameters to optimize
    lr = trial.suggest_float("ppo_learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_int("ppo_n_steps", 32, 2048, log=True)
    batch_size = trial.suggest_int("ppo_batch_size", 32, 256, log=True)
    gamma = trial.suggest_float("ppo_gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("ppo_gae_lambda", 0.9, 1.0)
    n_epochs = trial.suggest_int("ppo_n_epochs", 3, 30)
    ent_coef = trial.suggest_float("ppo_ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_float("ppo_clip_range", 0.1, 0.4)
    
    # Create environment
    env = create_env()
    
    # Create PPO model with the hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0
    )
    
    # Create evaluation callback with pause feature
    eval_env = create_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, f'trial_{trial.number}'),
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Add pause callback
    pause_callback = PauseCallback(reward_threshold=475.0, check_freq=1000)
    
    try:
        # Train the model with both callbacks
        model.learn(total_timesteps=50000, callback=[eval_callback, pause_callback])
        
        # Evaluate the trained model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        
        # Clean up
        env.close()
        eval_env.close()
        
        # Optuna maximizes, so we return the positive reward for CartPole
        return mean_reward
    
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        eval_env.close()
        return float('-inf')  # Return a low value to indicate failure

def optimize_a2c(trial: optuna.Trial) -> float:
    """Optimize A2C hyperparameters for CartPole-v1."""
    # Hyperparameters to optimize
    lr = trial.suggest_float("a2c_learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("a2c_n_steps", 1, 32)
    gamma = trial.suggest_float("a2c_gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("a2c_gae_lambda", 0.9, 1.0)
    ent_coef = trial.suggest_float("a2c_ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("a2c_vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("a2c_max_grad_norm", 0.1, 2.0)
    
    # Create environment
    env = create_env()
    
    # Create A2C model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=0
    )
    
    # Create evaluation callback
    eval_env = create_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, f'trial_{trial.number}'),
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Add pause callback
    pause_callback = PauseCallback(reward_threshold=475.0, check_freq=1000)
    
    try:
        # Train the model with both callbacks
        model.learn(total_timesteps=50000, callback=[eval_callback, pause_callback])
        
        # Evaluate the trained model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        
        # Clean up
        env.close()
        eval_env.close()
        
        # Optuna maximizes, so we return positive reward
        return mean_reward
    
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        eval_env.close()
        return float('-inf')  # Return a low value to indicate failure

def objective(trial: optuna.Trial) -> float:
    """Objective function to optimize RL algorithms for CartPole-v1."""
    # Choose which algorithm to optimize
    algorithm = trial.suggest_categorical("algorithm", ["PPO", "A2C"])
    
    if algorithm == "PPO":
        return optimize_ppo(trial)
    else:
        return optimize_a2c(trial)

def run_hpo(n_trials: int = 20) -> optuna.study.Study:
    """Run hyperparameter optimization."""
    study = optuna.create_study(direction="maximize")  # We want to maximize reward for CartPole
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    
    # Print the best trial information
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot the optimization history
    plt.figure(figsize=(10, 6))
    plot_optimization_history(study)
    plt.savefig(os.path.join(log_dir, "optimization_history.png"))
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    plot_param_importances(study)
    plt.savefig(os.path.join(log_dir, "param_importance.png"))
    
    return study

def push_to_github(repo_name, username=None):
    """Push the project to GitHub repository."""
    try:
        # Get the current working directory (project root)
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if .git directory exists
        if not os.path.exists(os.path.join(project_dir, ".git")):
            print("Initializing Git repository...")
            subprocess.run(["git", "init"], cwd=project_dir, check=True)
            
            # Configure GitHub username if provided
            if username:
                subprocess.run(["git", "config", "user.name", username], cwd=project_dir, check=True)
                
            # Create .gitignore file
            with open(os.path.join(project_dir, ".gitignore"), "w") as f:
                f.write("__pycache__/\n*.py[cod]\n*$py.class\n.env\n.venv\nenv/\nvenv/\nENV/\nenv.bak/\nvenv.bak/\n")
            
            # Add files to git
            subprocess.run(["git", "add", "."], cwd=project_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: RL HPO for CartPole"], cwd=project_dir, check=True)
            
            # Add GitHub remote
            remote_url = f"https://github.com/{username}/{repo_name}.git" if username else repo_name
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=project_dir, check=True)
            
            # Push to GitHub
            print(f"Pushing to GitHub repository: {remote_url}")
            subprocess.run(["git", "push", "-u", "origin", "master"], cwd=project_dir, check=True)
            print("Successfully pushed to GitHub!")
        else:
            # Repository already exists, just update it
            print("Updating existing Git repository...")
            subprocess.run(["git", "add", "."], cwd=project_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Update: RL HPO training results"], cwd=project_dir, check=True)
            subprocess.run(["git", "push"], cwd=project_dir, check=True)
            print("Successfully updated GitHub repository!")
            
        return True
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        return False

if __name__ == "__main__":
    # Run HPO with fewer trials for quicker results
    print("Starting hyperparameter optimization for CartPole-v1...")
    study = run_hpo(n_trials=10)
    
    # Save the study results
    study_file = os.path.join(log_dir, "study.pkl")
    print(f"Saving study results to {study_file}")
    
    try:
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
    except Exception as e:
        print(f"Error saving study: {e}")
    
    # Train a final model using the best hyperparameters
    print("Training final model with the best hyperparameters...")
    best_params = study.best_params.copy()
    
    # Extract the algorithm
    algorithm = best_params.pop("algorithm")
    
    # Create environment
    env = create_env()
    
    # Create pause callback for final model training
    final_pause_callback = PauseCallback(reward_threshold=475.0, check_freq=1000)
    
    # Create model with the best hyperparameters
    if algorithm == "PPO":
        # Filter out parameters that don't belong to PPO
        ppo_params = {k.replace("ppo_", ""): v for k, v in best_params.items() if k.startswith("ppo_")}
        model = PPO("MlpPolicy", env, **ppo_params, verbose=1)
    else:  # A2C
        # Filter out parameters that don't belong to A2C
        a2c_params = {k.replace("a2c_", ""): v for k, v in best_params.items() if k.startswith("a2c_")}
        model = A2C("MlpPolicy", env, **a2c_params, verbose=1)
    
    # Train the final model with pause callback
    model.learn(total_timesteps=50000, callback=final_pause_callback)
    
    # Save the final model
    model.save(os.path.join(models_dir, "best_model"))
    
    # Evaluate the final model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    print(f"Final model mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Clean up
    env.close()
    
    # Ask if user wants to push to GitHub
    print("\nDo you want to push this project to GitHub? (yes/no)")
    response = input().strip().lower()
    if response in ['yes', 'y']:
        print("Enter your GitHub username:")
        username = input().strip()
        print("Enter the repository name:")
        repo_name = input().strip()
        push_to_github(repo_name, username)
    else:
        print("Skipping GitHub upload.")