"""
Analyzer module for Rainbow DQN HPO project.
Visualizes and compares results from different HPO methods.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/analyzer.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Analyzer:
    """Analyzer for HPO results of Rainbow DQN agents."""
    
    def __init__(self, results_dir: str = "models"):
        """
        Initialize the Analyzer.
        
        Args:
            results_dir: Directory containing HPO results
        """
        self.results_dir = results_dir
        self.results = {}
        self.methods = []
        
        logger.info(f"Analyzer initialized with results_dir={results_dir}")
    
    def load_results(self, method_dirs: Optional[List[str]] = None) -> None:
        """
        Load HPO results from directories.
        
        Args:
            method_dirs: List of method directories to load, if None, load all subdirectories
        """
        if method_dirs is None:
            # Find all subdirectories in results_dir
            method_dirs = [d for d in os.listdir(self.results_dir) 
                          if os.path.isdir(os.path.join(self.results_dir, d))]
        
        for method_dir in method_dirs:
            dir_path = os.path.join(self.results_dir, method_dir)
            results_file = os.path.join(dir_path, "results.json")
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    
                    # Extract method name
                    method = results.get("method", method_dir)
                    self.results[method] = results
                    self.methods.append(method)
                    
                    logger.info(f"Loaded results for {method} from {results_file}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error loading {results_file}: {str(e)}")
            else:
                logger.warning(f"Results file not found in {dir_path}")
        
        if not self.results:
            logger.warning("No results loaded")
    
    def get_best_hyperparams(self, method: str) -> Dict[str, Any]:
        """
        Get best hyperparameters for a method.
        
        Args:
            method: Name of the method
            
        Returns:
            Dictionary of best hyperparameters
        """
        if method not in self.results:
            logger.warning(f"Method {method} not found in results")
            return {}
        
        return self.results[method].get("best_params", {})
    
    def compare_best_scores(self) -> pd.DataFrame:
        """
        Compare best scores across methods.
        
        Returns:
            DataFrame comparing best scores
        """
        data = []
        
        for method, results in self.results.items():
            best_score = results.get("best_score", float('nan'))
            runtime = results.get("runtime", float('nan'))
            
            data.append({
                "Method": method,
                "Best Score": best_score,
                "Runtime (s)": runtime,
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_learning_curves(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Plot learning curves for all methods.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure, if None, don't save
        """
        plt.figure(figsize=figsize)
        
        for method, results in self.results.items():
            # Extract trial data
            if method.startswith("Bayesian"):
                all_trials = results.get("all_trials", [])
                x = list(range(len(all_trials)))
                y = [trial.get("value", float('nan')) for trial in all_trials]
                plt.plot(x, y, 'o-', label=f"{method}")
            elif method.startswith("Evolution"):
                gen_stats = results.get("gen_stats", [])
                if gen_stats:
                    x = [stat.get("generation", i) for i, stat in enumerate(gen_stats)]
                    y_max = [stat.get("max", float('nan')) for stat in gen_stats]
                    y_avg = [stat.get("avg", float('nan')) for stat in gen_stats]
                    plt.plot(x, y_max, 'o-', label=f"{method} (Best)")
                    plt.plot(x, y_avg, 's--', label=f"{method} (Avg)")
            
        plt.xlabel('Trial / Generation')
        plt.ylabel('Reward')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_method_comparison(self, metrics: List[str] = ['Best Score', 'Runtime (s)'], 
                              figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot bar charts comparing methods.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
            save_path: Path to save the figure, if None, don't save
        """
        comparison_df = self.compare_best_scores()
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                sns.barplot(x='Method', y=metric, data=comparison_df, ax=axes[i])
                axes[i].set_title(f'Comparison of {metric}')
                axes[i].set_xlabel('Method')
                axes[i].set_ylabel(metric)
                
                # Add values on top of bars
                for p in axes[i].patches:
                    axes[i].annotate(f"{p.get_height():.2f}", 
                                    (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='bottom')
            else:
                logger.warning(f"Metric {metric} not found in results")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Method comparison saved to {save_path}")
        
        plt.show()
    
    def plot_hyperparameter_importance(self, figsize: Tuple[int, int] = (12, 6), 
                                     save_path: Optional[str] = None) -> None:
        """
        Plot hyperparameter importance.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure, if None, don't save
        """
        # Check if we have Bayesian Optimization results
        bayesian_methods = [m for m in self.methods if m.startswith("Bayesian")]
        
        if not bayesian_methods:
            logger.warning("No Bayesian Optimization results found for parameter importance")
            return
        
        plt.figure(figsize=figsize)
        
        for method in bayesian_methods:
            results = self.results[method]
            param_importances = results.get("param_importances", {})
            
            if param_importances:
                # Sort parameters by importance
                params = list(param_importances.keys())
                importances = list(param_importances.values())
                sorted_indices = np.argsort(importances)
                
                plt.barh([params[i] for i in sorted_indices], 
                        [importances[i] for i in sorted_indices])
                plt.xlabel('Importance')
                plt.ylabel('Hyperparameter')
                plt.title(f'Hyperparameter Importance for {method}')
                
                # Add values next to bars
                for i, v in enumerate([importances[j] for j in sorted_indices]):
                    plt.text(v + 0.01, i, f"{v:.3f}")
            else:
                logger.warning(f"No parameter importance data found for {method}")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Hyperparameter importance saved to {save_path}")
        
        plt.show()
    
    def plot_reward_distributions(self, figsize: Tuple[int, int] = (12, 6), 
                                save_path: Optional[str] = None) -> None:
        """
        Plot distributions of rewards for the best configurations.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure, if None, don't save
        """
        plt.figure(figsize=figsize)
        
        data = []
        
        for method, results in self.results.items():
            all_results = results.get("all_results", [])
            
            # Get rewards for best hyperparameters
            best_result = None
            for res in all_results:
                if res.get("hyperparams", {}) == results.get("best_params", {}):
                    best_result = res
                    break
            
            if best_result and "detailed_metrics" in best_result:
                detailed_metrics = best_result["detailed_metrics"]
                if "aggregate" in detailed_metrics:
                    rewards_per_seed = detailed_metrics["aggregate"].get("rewards_per_seed", [])
                    
                    for reward in rewards_per_seed:
                        data.append({
                            "Method": method,
                            "Reward": reward
                        })
        
        if data:
            df = pd.DataFrame(data)
            sns.violinplot(x="Method", y="Reward", data=df)
            plt.title("Reward Distributions for Best Configurations")
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Reward distributions saved to {save_path}")
            
            plt.show()
        else:
            logger.warning("No reward distribution data found")
    
    def plot_hyperparameter_parallel_coordinates(self, figsize: Tuple[int, int] = (14, 8), 
                                              save_path: Optional[str] = None) -> None:
        """
        Plot parallel coordinates plot for hyperparameters.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure, if None, don't save
        """
        # Collect all configurations and their scores
        data = []
        
        for method, results in self.results.items():
            all_results = results.get("all_results", [])
            
            for res in all_results:
                hyperparams = res.get("hyperparams", {})
                mean_reward = res.get("mean_reward", float('nan'))
                
                row = {
                    "Method": method,
                    "Mean Reward": mean_reward,
                    **hyperparams
                }
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            
            plt.figure(figsize=figsize)
            
            # Create a color map based on reward
            parallel_coordinates(df, "Method", colormap=plt.get_cmap("viridis"))
            plt.title("Hyperparameter Parallel Coordinates")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Hyperparameter parallel coordinates saved to {save_path}")
            
            plt.show()
        else:
            logger.warning("No hyperparameter data found for parallel coordinates")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a detailed comparison table.
        
        Returns:
            DataFrame with detailed comparison
        """
        data = []
        
        for method, results in self.results.items():
            best_score = results.get("best_score", float('nan'))
            runtime = results.get("runtime", float('nan'))
            total_trials = results.get("total_trials", 0)
            
            # Get detailed metrics for the best configuration
            all_results = results.get("all_results", [])
            best_result = None
            for res in all_results:
                if res.get("hyperparams", {}) == results.get("best_params", {}):
                    best_result = res
                    break
            
            std_reward = float('nan')
            variance = float('nan')
            steps_to_threshold = float('nan')
            
            if best_result and "detailed_metrics" in best_result:
                detailed_metrics = best_result["detailed_metrics"]
                if "aggregate" in detailed_metrics:
                    aggregate = detailed_metrics["aggregate"]
                    std_reward = aggregate.get("std_reward", float('nan'))
                    variance = aggregate.get("variance", float('nan'))
                    steps_to_threshold = aggregate.get("mean_steps_to_threshold", float('nan'))
            
            data.append({
                "Method": method,
                "Best Score": best_score,
                "Std Reward": std_reward,
                "Variance": variance,
                "Steps to Threshold": steps_to_threshold,
                "Runtime (s)": runtime,
                "Total Trials": total_trials
            })
        
        return pd.DataFrame(data)
    
    def generate_report(self, output_dir: str = "reports") -> str:
        """
        Generate a comprehensive report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"hpo_comparison_report_{timestamp}")
        
        # Create figures directory
        figures_dir = os.path.join(report_path, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generate comparison table
        comparison_df = self.create_comparison_table()
        comparison_table_path = os.path.join(report_path, "comparison_table.csv")
        comparison_df.to_csv(comparison_table_path, index=False)
        
        # Generate plots
        learning_curves_path = os.path.join(figures_dir, "learning_curves.png")
        self.plot_learning_curves(save_path=learning_curves_path)
        
        method_comparison_path = os.path.join(figures_dir, "method_comparison.png")
        self.plot_method_comparison(save_path=method_comparison_path)
        
        importance_path = os.path.join(figures_dir, "hyperparameter_importance.png")
        self.plot_hyperparameter_importance(save_path=importance_path)
        
        distributions_path = os.path.join(figures_dir, "reward_distributions.png")
        self.plot_reward_distributions(save_path=distributions_path)
        
        # Generate markdown report
        report_md_path = os.path.join(report_path, "report.md")
        
        with open(report_md_path, "w") as f:
            f.write("# Rainbow DQN Hyperparameter Optimization Comparison Report\n\n")
            
            f.write("## Comparison of Methods\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Learning Curves\n\n")
            f.write(f"![Learning Curves](figures/learning_curves.png)\n\n")
            
            f.write("## Method Comparison\n\n")
            f.write(f"![Method Comparison](figures/method_comparison.png)\n\n")
            
            f.write("## Hyperparameter Importance\n\n")
            f.write(f"![Hyperparameter Importance](figures/hyperparameter_importance.png)\n\n")
            
            f.write("## Reward Distributions\n\n")
            f.write(f"![Reward Distributions](figures/reward_distributions.png)\n\n")
            
            f.write("## Best Hyperparameters\n\n")
            for method in self.methods:
                f.write(f"### {method}\n\n")
                best_params = self.get_best_hyperparams(method)
                f.write("```json\n")
                f.write(json.dumps(best_params, indent=2))
                f.write("\n```\n\n")
        
        logger.info(f"Generated report at {report_path}")
        
        return report_path


# Helper functions for plotting
def parallel_coordinates(data, class_column, cols=None, ax=None, color=None, 
                         use_columns=False, xticks=None, colormap=None, **kwargs):
    """
    Parallel coordinates plot implementation for Pandas.
    
    Args:
        data: DataFrame containing the data
        class_column: Name of the column containing class labels
        cols: Columns to use, if None, use all columns
        ax: Matplotlib axis, if None, create a new one
        color: Color to use, if None, use colormap
        use_columns: Whether to use columns as index or not
        xticks: Custom x-ticks
        colormap: Matplotlib colormap
        **kwargs: Additional arguments for plotting
    
    Returns:
        Matplotlib axis
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    
    n = len(data)
    class_col_values = data[class_column]
    class_col_values_unique = class_col_values.unique()
    
    if cols is None:
        cols = data.columns.tolist()
        cols.remove(class_column)
    
    if ax is None:
        ax = plt.gca()
    
    x = list(range(len(cols)))
    
    if use_columns:
        x = cols
    
    # Create the plot
    for i, cls in enumerate(class_col_values_unique):
        # Get data for this class
        data_cls = data[class_col_values == cls].drop(class_column, axis=1)
        data_cls = data_cls[cols]
        
        # Normalize data for plotting
        min_max = {}
        for col in cols:
            min_max[col] = [data[col].min(), data[col].max()]
        
        for i in range(n):
            if class_col_values.iloc[i] != cls:
                continue
            row = data.drop(class_column, axis=1).iloc[i]
            row = row[cols]
            y = [(row[col] - min_max[col][0]) / (min_max[col][1] - min_max[col][0]) 
                for col in cols]
            
            if colormap is not None:
                color = colormap(i / n)
            
            ax.plot(x, y, color=color, label=cls, **kwargs)
    
    # Set the x-axis ticks
    if xticks is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(cols)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.grid(True)
    
    return ax


if __name__ == "__main__":
    # Simple test for analyzer
    analyzer = Analyzer(results_dir="models")
    print("Analyzer initialized")