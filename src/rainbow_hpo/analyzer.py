"""
Analyzer module for processing and comparing experiment results.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import glob
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path

from .utils.visualization import (
    plot_training_curve, 
    plot_loss_curve, 
    plot_hyperparameter_comparison, 
    plot_parameter_importance,
    create_comparison_report
)

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Class for analyzing reinforcement learning experiment results."""
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize the analyzer.
        
        Args:
            base_dir: Base directory containing experiment results
        """
        self.base_dir = base_dir
        self.trials_data = {}
        self.loaded_trials = []
        
    def load_trial(self, trial_id: int) -> Dict[str, Any]:
        """
        Load data for a specific trial.
        
        Args:
            trial_id: Trial ID
            
        Returns:
            Dictionary with trial data
        """
        trial_dir = os.path.join(self.base_dir, f"trial_{trial_id}")
        
        if not os.path.exists(trial_dir):
            logger.error(f"Trial directory not found: {trial_dir}")
            return {}
            
        # Load hyperparameters
        hyperparams_path = os.path.join(trial_dir, "hyperparameters.json")
        hyperparams = {}
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                hyperparams = json.load(f)
                
        # Load early stopping info
        early_stopping_path = os.path.join(trial_dir, "early_stopping_info.json")
        early_stopping_info = {}
        if os.path.exists(early_stopping_path):
            with open(early_stopping_path, "r") as f:
                early_stopping_info = json.load(f)
                
        # Load rewards and losses
        metrics_dir = os.path.join(trial_dir, "metrics")
        rewards = []
        losses = []
        
        rewards_path = os.path.join(metrics_dir, "rewards.csv")
        if os.path.exists(rewards_path):
            rewards_df = pd.read_csv(rewards_path)
            rewards = rewards_df["reward"].tolist()
            
        losses_path = os.path.join(metrics_dir, "losses.csv")
        if os.path.exists(losses_path):
            losses_df = pd.read_csv(losses_path)
            losses = losses_df["loss"].tolist()
            
        # Combine all data
        trial_data = {
            "trial_id": trial_id,
            "hyperparams": hyperparams,
            "early_stopping_info": early_stopping_info,
            "rewards": rewards,
            "losses": losses,
        }
        
        self.trials_data[trial_id] = trial_data
        self.loaded_trials.append(trial_id)
        
        logger.info(f"Loaded data for trial {trial_id}")
        return trial_data
    
    def load_all_trials(self) -> Dict[int, Dict[str, Any]]:
        """
        Load data for all available trials.
        
        Returns:
            Dictionary mapping trial IDs to trial data
        """
        # Find all trial directories
        trial_dirs = glob.glob(os.path.join(self.base_dir, "trial_*"))
        
        for trial_dir in trial_dirs:
            try:
                # Extract trial ID from directory name
                trial_id = int(os.path.basename(trial_dir).split("_")[1])
                self.load_trial(trial_id)
            except ValueError:
                # Skip directories that don't follow the trial_# naming convention
                continue
                
        logger.info(f"Loaded data for {len(self.loaded_trials)} trials")
        return self.trials_data
    
    def get_best_trial(self, metric: str = "best_eval_reward") -> Tuple[int, Dict[str, Any]]:
        """
        Find the best trial based on a given metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (best_trial_id, best_trial_data)
        """
        if not self.trials_data:
            logger.warning("No trials loaded. Loading all trials...")
            self.load_all_trials()
            
        best_trial_id = -1
        best_value = float('-inf')
        
        for trial_id, data in self.trials_data.items():
            # Check in early stopping info first
            if "early_stopping_info" in data and metric in data["early_stopping_info"]:
                value = data["early_stopping_info"][metric]
            # Then check directly in the data dictionary
            elif metric in data:
                value = data[metric]
            else:
                logger.warning(f"Metric {metric} not found for trial {trial_id}")
                continue
                
            if value > best_value:
                best_value = value
                best_trial_id = trial_id
                
        if best_trial_id == -1:
            logger.error(f"No trials found with metric {metric}")
            return -1, {}
            
        logger.info(f"Best trial by {metric}: {best_trial_id} with value {best_value}")
        return best_trial_id, self.trials_data[best_trial_id]
    
    def compare_trials(self, trial_ids: Optional[List[int]] = None, metric: str = "best_eval_reward") -> pd.DataFrame:
        """
        Compare multiple trials based on a given metric.
        
        Args:
            trial_ids: List of trial IDs to compare (None for all loaded trials)
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with comparison results
        """
        if trial_ids is None:
            trial_ids = self.loaded_trials
            
        # Load trials if not already loaded
        for trial_id in trial_ids:
            if trial_id not in self.trials_data:
                self.load_trial(trial_id)
                
        # Prepare comparison data
        comparison_data = []
        
        for trial_id in trial_ids:
            if trial_id not in self.trials_data:
                logger.warning(f"Trial {trial_id} not found")
                continue
                
            data = self.trials_data[trial_id]
            
            # Extract relevant metrics
            if "early_stopping_info" in data:
                early_info = data["early_stopping_info"]
                best_reward = early_info.get("best_eval_reward", None)
                final_reward = early_info.get("final_eval_reward", None)
                training_time = early_info.get("training_time_seconds", None)
                completed_episodes = early_info.get("completed_episodes", None)
            else:
                best_reward = None
                final_reward = None
                training_time = None
                completed_episodes = None
                
            # Calculate training stats if rewards are available
            if "rewards" in data and data["rewards"]:
                rewards = data["rewards"]
                mean_reward = np.mean(rewards)
                max_reward = np.max(rewards)
                min_reward = np.min(rewards)
                final_training_reward = rewards[-1]
            else:
                mean_reward = None
                max_reward = None
                min_reward = None
                final_training_reward = None
                
            # Get hyperparameters
            hyperparams = data.get("hyperparams", {})
            
            # Create row data
            row_data = {
                "trial_id": trial_id,
                "best_eval_reward": best_reward,
                "final_eval_reward": final_reward,
                "mean_training_reward": mean_reward,
                "max_training_reward": max_reward,
                "min_training_reward": min_reward,
                "final_training_reward": final_training_reward,
                "training_time_seconds": training_time,
                "completed_episodes": completed_episodes
            }
            
            # Add hyperparameters
            for key, value in hyperparams.items():
                row_data[f"param_{key}"] = value
                
            comparison_data.append(row_data)
            
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric if available
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
            
        return df
    
    def create_trial_report(self, trial_id: int, output_dir: Optional[str] = None) -> str:
        """
        Create a detailed report for a single trial.
        
        Args:
            trial_id: Trial ID
            output_dir: Output directory (None for trial directory)
            
        Returns:
            Path to the generated report
        """
        if trial_id not in self.trials_data:
            self.load_trial(trial_id)
            
        if trial_id not in self.trials_data:
            logger.error(f"Could not load data for trial {trial_id}")
            return ""
            
        data = self.trials_data[trial_id]
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, f"trial_{trial_id}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots if rewards available
        if "rewards" in data and data["rewards"]:
            rewards_plot_path = os.path.join(output_dir, "rewards_plot.png")
            plot_training_curve(
                data["rewards"], 
                title=f"Training Rewards (Trial {trial_id})",
                save_path=rewards_plot_path
            )
            
        # Create loss plot if losses available
        if "losses" in data and data["losses"]:
            losses_plot_path = os.path.join(output_dir, "losses_plot.png")
            plot_loss_curve(
                data["losses"], 
                title=f"Training Losses (Trial {trial_id})",
                save_path=losses_plot_path
            )
            
        # Generate report HTML
        report_path = os.path.join(output_dir, f"trial_{trial_id}_report.html")
        
        with open(report_path, "w") as f:
            f.write("<html><head>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }\n")
            f.write("h1, h2, h3 { color: #333; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
            f.write("th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("img { max-width: 100%; height: auto; margin: 20px 0; }\n")
            f.write("</style>\n")
            f.write("</head><body>\n")
            
            # Header
            f.write(f"<h1>Trial {trial_id} Report</h1>\n")
            
            # Performance summary
            f.write("<h2>Performance Summary</h2>\n")
            f.write("<table>\n")
            
            # Extract metrics from early stopping info
            if "early_stopping_info" in data:
                info = data["early_stopping_info"]
                early_metrics = [
                    ("Best Evaluation Reward", info.get("best_eval_reward", "N/A")),
                    ("Final Evaluation Reward", info.get("final_eval_reward", "N/A")),
                    ("Training Time (seconds)", info.get("training_time_seconds", "N/A")),
                    ("Completed Episodes", info.get("completed_episodes", "N/A")),
                    ("Early Stopping Threshold Reached", "Yes" if info.get("threshold_reached", False) else "No"),
                    ("Early Stopping Threshold", info.get("early_stopping_threshold", "N/A")),
                    ("Patience", info.get("patience", "N/A"))
                ]
                
                for name, value in early_metrics:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    f.write(f"<tr><td>{name}</td><td>{formatted_value}</td></tr>\n")
            
            # Add training metrics
            if "rewards" in data and data["rewards"]:
                rewards = data["rewards"]
                training_metrics = [
                    ("Mean Training Reward", f"{np.mean(rewards):.4f}"),
                    ("Max Training Reward", f"{np.max(rewards):.4f}"),
                    ("Min Training Reward", f"{np.min(rewards):.4f}"),
                    ("Final Training Reward", f"{rewards[-1]:.4f}"),
                    ("Total Training Episodes", len(rewards))
                ]
                
                for name, value in training_metrics:
                    f.write(f"<tr><td>{name}</td><td>{value}</td></tr>\n")
                    
            f.write("</table>\n")
            
            # Hyperparameters
            f.write("<h2>Hyperparameters</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Parameter</th><th>Value</th></tr>\n")
            
            if "hyperparams" in data:
                for param, value in data["hyperparams"].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        formatted_value = f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    f.write(f"<tr><td>{param}</td><td>{formatted_value}</td></tr>\n")
                    
            f.write("</table>\n")
            
            # Plots
            f.write("<h2>Training Plots</h2>\n")
            
            if "rewards" in data and data["rewards"]:
                f.write("<h3>Rewards</h3>\n")
                f.write(f'<img src="rewards_plot.png" alt="Training Rewards">\n')
                
            if "losses" in data and data["losses"]:
                f.write("<h3>Losses</h3>\n")
                f.write(f'<img src="losses_plot.png" alt="Training Losses">\n')
                
            f.write("</body></html>\n")
            
        logger.info(f"Trial report generated at {report_path}")
        return report_path
    
    def create_comparison_report(self, trial_ids: Optional[List[int]] = None, 
                                output_path: Optional[str] = None) -> str:
        """
        Create a comparison report for multiple trials.
        
        Args:
            trial_ids: List of trial IDs to compare (None for all loaded trials)
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if trial_ids is None:
            if not self.loaded_trials:
                self.load_all_trials()
            trial_ids = self.loaded_trials
            
        # Ensure all trials are loaded
        for trial_id in trial_ids:
            if trial_id not in self.trials_data:
                self.load_trial(trial_id)
                
        # Get default output path if not specified
        if output_path is None:
            output_path = os.path.join(self.base_dir, "trial_comparison_report.html")
            
        # Create comparison report using the visualization utility
        return create_comparison_report(self.base_dir, trial_ids, output_path)
    
    def analyze_hyperparameter_importance(self, metric: str = "best_eval_reward") -> Dict[str, float]:
        """
        Analyze the importance of different hyperparameters.
        
        Args:
            metric: Metric to use for analysis
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not self.trials_data:
            self.load_all_trials()
            
        # Get comparison DataFrame
        df = self.compare_trials(metric=metric)
        
        # Identify hyperparameter columns
        hyperparam_cols = [col for col in df.columns if col.startswith("param_")]
        
        if not hyperparam_cols:
            logger.warning("No hyperparameters found in trial data")
            return {}
            
        # Calculate correlation between hyperparameters and metric
        importance_scores = {}
        
        for param in hyperparam_cols:
            # Skip non-numeric parameters
            if not pd.api.types.is_numeric_dtype(df[param]):
                continue
                
            # Calculate correlation
            if metric in df.columns:
                correlation = df[param].corr(df[metric])
                importance_scores[param.replace("param_", "")] = abs(correlation)
                
        logger.info(f"Analyzed importance of {len(importance_scores)} hyperparameters")
        return importance_scores
    
    def plot_hyperparameter_importance(self, output_path: Optional[str] = None) -> str:
        """
        Plot hyperparameter importance.
        
        Args:
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        importance_scores = self.analyze_hyperparameter_importance()
        
        if not importance_scores:
            logger.warning("No hyperparameter importance scores to plot")
            return ""
            
        if output_path is None:
            output_path = os.path.join(self.base_dir, "hyperparameter_importance.png")
            
        plot_parameter_importance(
            importance_scores,
            title="Hyperparameter Importance",
            save_path=output_path
        )
        
        logger.info(f"Hyperparameter importance plot saved to {output_path}")
        return output_path


def analyze_experiment(base_dir: str = "models", output_dir: Optional[str] = None):
    """
    Convenience function to analyze an experiment and generate reports.
    
    Args:
        base_dir: Base directory containing experiment results
        output_dir: Output directory for reports
    """
    analyzer = ExperimentAnalyzer(base_dir=base_dir)
    
    # Load all trial data
    analyzer.load_all_trials()
    
    if not analyzer.loaded_trials:
        logger.error(f"No trial data found in {base_dir}")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, "analysis")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best trial
    best_trial_id, _ = analyzer.get_best_trial()
    
    # Generate individual reports for each trial
    for trial_id in analyzer.loaded_trials:
        analyzer.create_trial_report(trial_id, output_dir=os.path.join(output_dir, f"trial_{trial_id}"))
        
    # Generate comparison report
    analyzer.create_comparison_report(output_path=os.path.join(output_dir, "trial_comparison.html"))
    
    # Generate hyperparameter importance plot
    analyzer.plot_hyperparameter_importance(output_path=os.path.join(output_dir, "hyperparameter_importance.png"))
    
    logger.info(f"Experiment analysis completed. Results saved to {output_dir}")
    logger.info(f"Best trial: {best_trial_id}")