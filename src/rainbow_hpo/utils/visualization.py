"""
Visualization utilities for Rainbow DQN experiments.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# Set the style for all plots
sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.size': 12})


def plot_training_curve(rewards: List[float], title: str = "Training Curve", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the training curve showing episode rewards over time.
    
    Args:
        rewards: List of episode rewards
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = range(len(rewards))
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot smoothed rewards
    window_size = min(100, max(1, len(rewards) // 10))
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size).mean()
    ax.plot(episodes, smoothed_rewards, linewidth=2, color='blue', label=f'Smoothed (window={window_size})')
    
    # Add horizontal lines for min, mean, max
    ax.axhline(y=np.max(rewards), color='green', linestyle='--', alpha=0.7, label=f'Max: {np.max(rewards):.2f}')
    ax.axhline(y=np.mean(rewards), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(rewards):.2f}')
    ax.axhline(y=np.min(rewards), color='orange', linestyle='--', alpha=0.7, label=f'Min: {np.min(rewards):.2f}')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training curve plot saved to {save_path}")
    
    return fig


def plot_loss_curve(losses: List[float], title: str = "Loss Curve", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the training loss curve.
    
    Args:
        losses: List of training losses
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = range(len(losses))
    
    # For large number of steps, we need to smooth the loss curve
    if len(losses) > 1000:
        window_size = len(losses) // 100
        smoothed_losses = pd.Series(losses).rolling(window=window_size).mean()
        ax.plot(steps, smoothed_losses, linewidth=2, color='red', label=f'Smoothed Loss (window={window_size})')
        # Plot raw losses with low alpha
        ax.plot(steps, losses, alpha=0.1, color='red', label='Raw Loss')
    else:
        ax.plot(steps, losses, linewidth=1, color='red', label='Loss')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Loss curve plot saved to {save_path}")
    
    return fig


def plot_hyperparameter_comparison(results: Dict[str, Dict[str, Any]], metric: str = "best_eval_reward", 
                                   title: str = "Hyperparameter Comparison", 
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple hyperparameter configurations.
    
    Args:
        results: Dictionary mapping trial IDs to result dictionaries
        metric: Metric to compare
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract metric values
    trial_ids = list(results.keys())
    metric_values = [results[tid].get(metric, 0) for tid in trial_ids]
    
    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    trial_ids = [trial_ids[i] for i in sorted_indices]
    metric_values = [metric_values[i] for i in sorted_indices]
    
    # Create bar chart
    bars = ax.barh(trial_ids, metric_values)
    
    # Add value labels to the right of each bar
    for i, v in enumerate(metric_values):
        ax.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    # Customization
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Hyperparameter comparison plot saved to {save_path}")
    
    return fig


def plot_parameter_importance(parameter_importances: Dict[str, float], 
                             title: str = "Parameter Importance", 
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot parameter importance based on optimization results.
    
    Args:
        parameter_importances: Dictionary mapping parameter names to importance scores
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort parameters by importance
    sorted_params = sorted(parameter_importances.items(), key=lambda x: x[1], reverse=True)
    param_names = [x[0] for x in sorted_params]
    importance_values = [x[1] for x in sorted_params]
    
    # Create bar chart
    bars = ax.barh(param_names, importance_values)
    
    # Add value labels
    for i, v in enumerate(importance_values):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Parameter importance plot saved to {save_path}")
    
    return fig


def create_comparison_report(base_dir: str, trials: List[int], output_path: Optional[str] = None) -> str:
    """
    Create a comparison report for multiple trials.
    
    Args:
        base_dir: Base directory containing trial results
        trials: List of trial IDs to compare
        output_path: Path to save the report HTML
        
    Returns:
        Path to the generated report
    """
    import pandas as pd
    from pathlib import Path
    
    # Collect trial data
    trial_data = []
    
    for trial_id in trials:
        trial_dir = os.path.join(base_dir, f"trial_{trial_id}")
        
        # Check if trial directory exists
        if not os.path.exists(trial_dir):
            logger.warning(f"Trial directory not found: {trial_dir}")
            continue
        
        # Load hyperparameters
        hyperparams_path = os.path.join(trial_dir, "hyperparameters.json")
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                hyperparams = json.load(f)
        else:
            hyperparams = {}
        
        # Load early stopping info if available
        early_stopping_path = os.path.join(trial_dir, "early_stopping_info.json")
        if os.path.exists(early_stopping_path):
            with open(early_stopping_path, "r") as f:
                early_stopping_info = json.load(f)
        else:
            early_stopping_info = {}
        
        # Load rewards if available
        metrics_dir = os.path.join(trial_dir, "metrics")
        rewards_path = os.path.join(metrics_dir, "rewards.csv")
        if os.path.exists(rewards_path):
            rewards_df = pd.read_csv(rewards_path)
            final_reward = rewards_df["reward"].iloc[-1]
            mean_reward = rewards_df["reward"].mean()
            max_reward = rewards_df["reward"].max()
        else:
            final_reward = None
            mean_reward = None
            max_reward = None
        
        # Combine all data
        trial_info = {
            "trial_id": trial_id,
            "final_reward": final_reward,
            "mean_reward": mean_reward,
            "max_reward": max_reward,
            "best_eval_reward": early_stopping_info.get("best_eval_reward"),
            "final_eval_reward": early_stopping_info.get("final_eval_reward"),
            "completed_episodes": early_stopping_info.get("completed_episodes"),
            "training_time_seconds": early_stopping_info.get("training_time_seconds")
        }
        
        # Add hyperparameters
        for key, value in hyperparams.items():
            trial_info[f"param_{key}"] = value
        
        trial_data.append(trial_info)
    
    # Create dataframe
    if not trial_data:
        logger.error("No valid trial data found")
        return None
    
    df = pd.DataFrame(trial_data)
    
    # Sort by best_eval_reward, descending
    if "best_eval_reward" in df.columns:
        df = df.sort_values("best_eval_reward", ascending=False)
    
    # Generate HTML report
    html_content = []
    html_content.append("<html><head>")
    html_content.append("<style>")
    html_content.append("body { font-family: Arial, sans-serif; margin: 40px; }")
    html_content.append("table { border-collapse: collapse; width: 100%; margin-top: 20px; }")
    html_content.append("th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }")
    html_content.append("th { background-color: #f2f2f2; }")
    html_content.append("tr:hover {background-color: #f5f5f5;}")
    html_content.append("h1, h2 { color: #333; }")
    html_content.append(".best-row { background-color: #e6ffe6; }")
    html_content.append("</style>")
    html_content.append("</head><body>")
    
    html_content.append(f"<h1>Trial Comparison Report</h1>")
    html_content.append(f"<p>Comparing {len(trial_data)} trials</p>")
    
    # Top trials by evaluation reward
    html_content.append("<h2>Trials Ranked by Evaluation Performance</h2>")
    html_content.append("<table>")
    
    # Table headers
    headers = ["Rank", "Trial ID", "Best Eval Reward", "Final Eval Reward", 
               "Mean Training Reward", "Max Training Reward", "Episodes", "Training Time (s)"]
    html_content.append("<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>")
    
    # Table rows
    for i, (_, row) in enumerate(df.iterrows()):
        row_class = " class='best-row'" if i == 0 else ""
        html_content.append(f"<tr{row_class}>")
        html_content.append(f"<td>{i+1}</td>")
        html_content.append(f"<td>{row['trial_id']}</td>")
        html_content.append(f"<td>{row['best_eval_reward']:.2f}</td>")
        html_content.append(f"<td>{row['final_eval_reward']:.2f}</td>")
        html_content.append(f"<td>{row['mean_reward']:.2f}</td>")
        html_content.append(f"<td>{row['max_reward']:.2f}</td>")
        html_content.append(f"<td>{row['completed_episodes']}</td>")
        html_content.append(f"<td>{row['training_time_seconds']:.1f}</td>")
        html_content.append("</tr>")
    
    html_content.append("</table>")
    
    # Hyperparameter comparison
    html_content.append("<h2>Hyperparameter Comparison</h2>")
    html_content.append("<table>")
    
    # Get all hyperparameter columns
    hyperparam_cols = [col for col in df.columns if col.startswith("param_")]
    
    # Table headers
    headers = ["Rank", "Trial ID", "Best Eval Reward"] + [col.replace("param_", "") for col in hyperparam_cols]
    html_content.append("<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>")
    
    # Table rows
    for i, (_, row) in enumerate(df.iterrows()):
        row_class = " class='best-row'" if i == 0 else ""
        html_content.append(f"<tr{row_class}>")
        html_content.append(f"<td>{i+1}</td>")
        html_content.append(f"<td>{row['trial_id']}</td>")
        html_content.append(f"<td>{row['best_eval_reward']:.2f}</td>")
        
        for col in hyperparam_cols:
            val = row[col]
            if isinstance(val, (int, float)):
                html_content.append(f"<td>{val}</td>")
            else:
                html_content.append(f"<td>{val}</td>")
                
        html_content.append("</tr>")
    
    html_content.append("</table>")
    html_content.append("</body></html>")
    
    # Write HTML file
    if not output_path:
        output_path = os.path.join(base_dir, "trial_comparison_report.html")
    
    with open(output_path, "w") as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Comparison report saved to {output_path}")
    return output_path