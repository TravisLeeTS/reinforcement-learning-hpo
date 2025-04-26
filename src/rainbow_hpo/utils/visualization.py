"""
Visualization tools for hyperparameter optimization.
Generates plots and visualizations for HPO results analysis.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# Check for visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib and/or seaborn not available. Visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    # Set plotly theme
    pio.templates.default = "plotly_white"
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive visualizations will be disabled.")


class HPOVisualizer:
    """
    Creates visualizations for hyperparameter optimization results.
    Supports both static (matplotlib) and interactive (plotly) visualizations.
    """
    
    def __init__(self, output_dir: str = "plots", 
                interactive: bool = True,
                static: bool = True,
                show_plots: bool = False,
                theme: str = "darkgrid"):
        """
        Initialize the HPO visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            interactive: Whether to generate interactive plotly visualizations
            static: Whether to generate static matplotlib visualizations
            show_plots: Whether to display plots in addition to saving
            theme: Theme for static plots (seaborn)
        """
        self.output_dir = output_dir
        self.interactive = interactive and PLOTLY_AVAILABLE
        self.static = static and PLOTTING_AVAILABLE
        self.show_plots = show_plots
        self.theme = theme
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up theme for static plots
        if self.static:
            try:
                sns.set_theme(style=self.theme)
            except:
                pass  # Older versions don't have set_theme
        
        if not self.interactive and not self.static:
            logger.warning("Neither interactive nor static visualizations are available. "
                          "Please install matplotlib, seaborn, and/or plotly.")
    
    def plot_optimization_history(self, results: List[Dict[str, Any]], 
                                title: str = "Optimization History",
                                filename: str = "optimization_history"):
        """
        Plot the optimization history showing the value of each trial.
        
        Args:
            results: List of dictionaries with trial results
            title: Plot title
            filename: Base filename for saving plots
        """
        if not results:
            logger.warning("No results provided for optimization history plot")
            return
        
        # Extract trial indices and values
        trial_indices = []
        values = []
        best_so_far = []
        
        current_best = float('-inf')
        for i, result in enumerate(results):
            value = result.get('value', float('-inf'))
            trial_id = result.get('trial_id', i)
            
            trial_indices.append(trial_id)
            values.append(value)
            
            # Update best seen so far
            current_best = max(current_best, value)
            best_so_far.append(current_best)
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot individual trial values
            ax.scatter(trial_indices, values, alpha=0.7, label="Trial Value")
            
            # Plot best value so far
            ax.plot(trial_indices, best_so_far, 'r-', label="Best So Far")
            
            # Add labels and title
            ax.set_xlabel("Trial")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static optimization history plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            df = pd.DataFrame({
                "Trial": trial_indices,
                "Value": values,
                "Best So Far": best_so_far
            })
            
            fig = go.Figure()
            
            # Add scatter plot for individual trial values
            fig.add_trace(go.Scatter(
                x=df["Trial"],
                y=df["Value"],
                mode="markers",
                name="Trial Value",
                marker=dict(size=8, opacity=0.7)
            ))
            
            # Add line plot for best value so far
            fig.add_trace(go.Scatter(
                x=df["Trial"],
                y=df["Best So Far"],
                mode="lines",
                name="Best So Far",
                line=dict(color="red", width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Trial",
                yaxis_title="Value",
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
                hovermode="closest"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive optimization history plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def plot_parameter_importance(self, importances: Dict[str, float],
                                title: str = "Parameter Importance",
                                filename: str = "parameter_importance"):
        """
        Plot parameter importance from optimization results.
        
        Args:
            importances: Dictionary mapping parameter names to importance values
            title: Plot title
            filename: Base filename for saving plots
        """
        if not importances:
            logger.warning("No parameter importances provided for plot")
            return
        
        # Sort parameters by importance (descending)
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        importance_values = [p[1] for p in sorted_params]
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal bar plot
            y_pos = range(len(param_names))
            ax.barh(y_pos, importance_values, align='center')
            
            # Add labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(param_names)
            ax.invert_yaxis()  # Highest values at the top
            ax.set_xlabel('Importance')
            ax.set_title(title)
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static parameter importance plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            df = pd.DataFrame({
                "Parameter": param_names,
                "Importance": importance_values
            })
            
            fig = px.bar(df, x="Importance", y="Parameter", orientation='h',
                       title=title)
            
            # Update layout
            fig.update_layout(
                yaxis=dict(
                    categoryorder='total ascending',  # Sort bars by value
                ),
                hovermode="closest"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive parameter importance plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def plot_comparison(self, method_results: Dict[str, List[float]],
                      title: str = "HPO Method Comparison",
                      filename: str = "method_comparison"):
        """
        Create a box plot comparing different HPO methods.
        
        Args:
            method_results: Dictionary mapping method names to lists of result values
            title: Plot title
            filename: Base filename for saving plots
        """
        if not method_results:
            logger.warning("No method results provided for comparison plot")
            return
        
        # Prepare data for plotting
        data = []
        method_names = []
        
        for method_name, values in method_results.items():
            if not values:
                continue
            data.append(values)
            method_names.append(method_name)
        
        if not data:
            logger.warning("No valid data for comparison plot")
            return
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot
            bp = ax.boxplot(data, patch_artist=True, labels=method_names)
            
            # Customize box colors
            colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i % len(colors)])
            
            # Add jittered points for individual values
            for i, values in enumerate(data):
                # Add some random x jitter
                x = [i+1 + np.random.normal(0, 0.05) for _ in range(len(values))]
                ax.scatter(x, values, alpha=0.5, s=20, c='black', zorder=3)
            
            # Add mean markers
            means = [np.mean(d) for d in data]
            ax.plot(range(1, len(means) + 1), means, 'rd', markeredgecolor='black', markersize=8)
            
            # Add labels and title
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.grid(alpha=0.3)
            
            # Add mean values as text above each box
            for i, mean in enumerate(means):
                ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static method comparison plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            # Prepare data for plotly
            fig = go.Figure()
            
            # Add box traces for each method
            colors = ['rgba(93, 164, 214, 0.7)', 'rgba(44, 160, 101, 0.7)', 
                    'rgba(255, 65, 54, 0.7)', 'rgba(207, 114, 255, 0.7)']
            
            for i, (method, values) in enumerate(zip(method_names, data)):
                color_idx = i % len(colors)
                
                # Add box plot
                fig.add_trace(go.Box(
                    y=values,
                    name=method,
                    marker_color=colors[color_idx],
                    boxmean=True,  # Add mean marker
                ))
                
                # Add individual points as a scatter
                fig.add_trace(go.Scatter(
                    y=values,
                    x=[method] * len(values),
                    mode='markers',
                    marker=dict(
                        color='rgba(0, 0, 0, 0.3)',
                        size=6,
                    ),
                    name=f"{method} points",
                    showlegend=False,
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                yaxis_title="Value",
                boxmode="group"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive method comparison plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def plot_learning_curves(self, learning_data: Dict[str, Dict[str, List[float]]],
                           title: str = "Learning Curves",
                           filename: str = "learning_curves",
                           x_label: str = "Episode",
                           y_label: str = "Reward"):
        """
        Plot learning curves for multiple methods or agents.
        
        Args:
            learning_data: Nested dictionary with method/agent names and their x, y values
                           Format: {"method1": {"x": [...], "y": [...], "std": [...]}, ...}
            title: Plot title
            filename: Base filename for saving plots
            x_label: Label for x-axis
            y_label: Label for y-axis
        """
        if not learning_data:
            logger.warning("No learning data provided for learning curves plot")
            return
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each learning curve
            for method_name, data in learning_data.items():
                x = data.get("x", range(len(data.get("y", []))))
                y = data.get("y", [])
                std = data.get("std", None)
                
                if not x or not y or len(x) != len(y):
                    logger.warning(f"Invalid data for {method_name} learning curve")
                    continue
                
                # Plot mean
                line, = ax.plot(x, y, label=method_name)
                
                # Plot standard deviation band if available
                if std is not None and len(std) == len(y):
                    ax.fill_between(x, 
                                  np.array(y) - np.array(std), 
                                  np.array(y) + np.array(std),
                                  alpha=0.2, color=line.get_color())
            
            # Add labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static learning curves plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            fig = go.Figure()
            
            # Add each learning curve
            for method_name, data in learning_data.items():
                x = data.get("x", range(len(data.get("y", []))))
                y = data.get("y", [])
                std = data.get("std", None)
                
                if not x or not y or len(x) != len(y):
                    logger.warning(f"Invalid data for {method_name} learning curve")
                    continue
                
                # Plot mean line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=method_name,
                ))
                
                # Add standard deviation bands if available
                if std is not None and len(std) == len(y):
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([np.array(y) + np.array(std), 
                                        (np.array(y) - np.array(std))[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,0,0,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                hovermode="closest"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive learning curves plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def plot_hyperparameter_correlation(self, trials: List[Dict[str, Any]],
                                      target_param: str,
                                      value_key: str = "value",
                                      title: str = None,
                                      filename: str = None):
        """
        Plot correlation between a hyperparameter and objective value.
        
        Args:
            trials: List of trial dictionaries with parameters and values
            target_param: Name of the hyperparameter to analyze
            value_key: Key for the objective value in trial dictionaries
            title: Plot title (defaults to "Effect of {param}")
            filename: Base filename (defaults to "correlation_{param}")
        """
        if not trials:
            logger.warning("No trials provided for hyperparameter correlation plot")
            return
        
        # Extract parameter values and objective values
        param_values = []
        objective_values = []
        
        for trial in trials:
            if target_param not in trial:
                continue
                
            param_values.append(trial[target_param])
            objective_values.append(trial.get(value_key, float('nan')))
        
        if not param_values:
            logger.warning(f"Parameter {target_param} not found in trials")
            return
        
        # Set default values if not provided
        if title is None:
            title = f"Effect of {target_param}"
        if filename is None:
            filename = f"correlation_{target_param}"
        
        # Check parameter type to determine plot type
        param_type = "categorical" if isinstance(param_values[0], str) else "numerical"
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if param_type == "numerical":
                # For numerical parameters, create scatter plot
                ax.scatter(param_values, objective_values, alpha=0.7)
                
                # Try to fit a trend line
                try:
                    z = np.polyfit(param_values, objective_values, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(param_values), p(sorted(param_values)), "r--", alpha=0.7)
                except:
                    pass  # Skip trend line if it can't be computed
                
                ax.set_xlabel(target_param)
                
            else:
                # For categorical parameters, create box plot
                df = pd.DataFrame({
                    "param": param_values,
                    "value": objective_values
                })
                
                # Group by parameter value and compute statistics
                grouped = df.groupby("param")["value"].agg(['mean', 'count']).reset_index()
                counts = {row['param']: row['count'] for _, row in grouped.iterrows()}
                
                # Create box plot
                sns.boxplot(x="param", y="value", data=df, ax=ax)
                
                # Add individual points
                sns.swarmplot(x="param", y="value", data=df, color=".25", ax=ax)
                
                # Add count to labels
                new_labels = [f"{label.get_text()}\n(n={counts.get(label.get_text(), 0)})" 
                            for label in ax.get_xticklabels()]
                ax.set_xticklabels(new_labels)
                
                ax.set_xlabel(target_param)
            
            # Common setup
            ax.set_ylabel("Objective Value")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static parameter correlation plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            if param_type == "numerical":
                # For numerical parameters, create scatter plot with trend line
                fig = px.scatter(
                    x=param_values,
                    y=objective_values,
                    trendline="ols" if len(param_values) > 2 else None,
                    labels={
                        "x": target_param,
                        "y": "Objective Value"
                    },
                    title=title
                )
                
            else:
                # For categorical parameters, create box plot
                df = pd.DataFrame({
                    "param": param_values,
                    "value": objective_values
                })
                
                fig = px.box(
                    df,
                    x="param",
                    y="value",
                    points="all",
                    labels={
                        "param": target_param,
                        "value": "Objective Value"
                    },
                    title=title
                )
                
                # Add count to hover
                fig.update_traces(
                    hovertemplate=(
                        f"{target_param}: %{{x}}<br>"
                        "Value: %{y}<br>"
                        "Count: %{customdata[0]}"
                    )
                )
            
            # Update layout
            fig.update_layout(
                hovermode="closest"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive parameter correlation plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def plot_resource_usage(self, resource_data: Dict[str, List],
                          title: str = "Resource Usage",
                          filename: str = "resource_usage"):
        """
        Plot resource usage over time.
        
        Args:
            resource_data: Dictionary with resource usage data
            title: Plot title
            filename: Base filename for saving plots
        """
        if not resource_data or "timestamps" not in resource_data:
            logger.warning("No valid resource data provided for resource usage plot")
            return
        
        timestamps = resource_data["timestamps"]
        if len(timestamps) < 2:
            logger.warning("Not enough data points for resource usage plot")
            return
        
        # Convert absolute timestamps to relative times (seconds from start)
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        # Generate static plot with matplotlib
        if self.static:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot CPU and memory percentages on first Y-axis
            if "cpu_percent" in resource_data:
                ax1.plot(rel_times, resource_data["cpu_percent"], 'b-', label='CPU %')
            if "memory_percent" in resource_data:
                ax1.plot(rel_times, resource_data["memory_percent"], 'g-', label='Memory %')
            if "gpu_utilization" in resource_data and resource_data["gpu_utilization"][0] is not None:
                ax1.plot(rel_times, resource_data["gpu_utilization"], 'm-', label='GPU %')
                
            # Set up first Y-axis
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Usage %')
            ax1.set_ylim(0, 100)
            
            # Plot memory usage on second Y-axis
            if "memory_used" in resource_data:
                memory_gb = [m/1024.0 for m in resource_data["memory_used"]]  # MB to GB
                ax2.plot(rel_times, memory_gb, 'r-', label='Memory (GB)')
            if "gpu_memory_used" in resource_data and resource_data["gpu_memory_used"][0] is not None:
                gpu_memory_gb = [m/1024.0 for m in resource_data["gpu_memory_used"]]  # MB to GB
                ax2.plot(rel_times, gpu_memory_gb, 'c-', label='GPU Memory (GB)')
                
            # Set up second Y-axis
            ax2.set_ylabel('Memory (GB)')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid and title
            ax1.grid(alpha=0.3)
            plt.title(title)
            
            # Save plot
            static_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.tight_layout()
            plt.savefig(static_path)
            logger.info(f"Saved static resource usage plot to {static_path}")
            
            if self.show_plots:
                plt.show()
            
            plt.close()
        
        # Generate interactive plot with plotly
        if self.interactive:
            fig = go.Figure()
            
            # Plot CPU and memory percentages
            if "cpu_percent" in resource_data:
                fig.add_trace(go.Scatter(
                    x=rel_times,
                    y=resource_data["cpu_percent"],
                    name="CPU %",
                    line=dict(color="blue", width=2)
                ))
                
            if "memory_percent" in resource_data:
                fig.add_trace(go.Scatter(
                    x=rel_times,
                    y=resource_data["memory_percent"],
                    name="Memory %",
                    line=dict(color="green", width=2)
                ))
                
            if "gpu_utilization" in resource_data and resource_data["gpu_utilization"][0] is not None:
                fig.add_trace(go.Scatter(
                    x=rel_times,
                    y=resource_data["gpu_utilization"],
                    name="GPU %",
                    line=dict(color="magenta", width=2)
                ))
            
            # Add memory usage on secondary y-axis
            if "memory_used" in resource_data:
                memory_gb = [m/1024.0 for m in resource_data["memory_used"]]  # MB to GB
                fig.add_trace(go.Scatter(
                    x=rel_times,
                    y=memory_gb,
                    name="Memory (GB)",
                    line=dict(color="red", width=2),
                    yaxis="y2"
                ))
                
            if "gpu_memory_used" in resource_data and resource_data["gpu_memory_used"][0] is not None:
                gpu_memory_gb = [m/1024.0 for m in resource_data["gpu_memory_used"]]  # MB to GB
                fig.add_trace(go.Scatter(
                    x=rel_times,
                    y=gpu_memory_gb,
                    name="GPU Memory (GB)",
                    line=dict(color="cyan", width=2),
                    yaxis="y2"
                ))
            
            # Update layout with double y-axis
            fig.update_layout(
                title=title,
                xaxis_title="Time (seconds)",
                yaxis=dict(
                    title="Usage %",
                    range=[0, 100],
                ),
                yaxis2=dict(
                    title="Memory (GB)",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode="closest"
            )
            
            # Save as interactive HTML
            interactive_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(interactive_path)
            logger.info(f"Saved interactive resource usage plot to {interactive_path}")
            
            if self.show_plots:
                fig.show()
    
    def generate_dashboard(self, data: Dict[str, Any], 
                         title: str = "HPO Dashboard", 
                         filename: str = "hpo_dashboard"):
        """
        Generate a comprehensive dashboard with all visualizations.
        
        Args:
            data: Dictionary with all necessary data
            title: Dashboard title
            filename: Base filename for saving the dashboard
        """
        if not self.interactive:
            logger.warning("Interactive plotting is required for dashboards")
            return
        
        try:
            import plotly.subplots as sp
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly is required for dashboard generation")
            return
        
        # Create dashboard layout with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Optimization History",
                "Parameter Importance",
                "Learning Curves",
                "Resource Usage"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Optimization History
        if "optimization_history" in data:
            history = data["optimization_history"]
            trial_indices = history.get("trial_indices", [])
            values = history.get("values", [])
            best_so_far = history.get("best_so_far", [])
            
            # Add individual trial values
            fig.add_trace(
                go.Scatter(
                    x=trial_indices,
                    y=values,
                    mode="markers",
                    name="Trial Value",
                    marker=dict(size=8, opacity=0.7)
                ),
                row=1, col=1
            )
            
            # Add best value so far
            fig.add_trace(
                go.Scatter(
                    x=trial_indices,
                    y=best_so_far,
                    mode="lines",
                    name="Best So Far",
                    line=dict(color="red", width=2)
                ),
                row=1, col=1
            )
        
        # 2. Parameter Importance
        if "parameter_importance" in data:
            importances = data["parameter_importance"]
            
            # Sort parameters by importance (descending)
            sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in sorted_params]
            importance_values = [p[1] for p in sorted_params]
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=param_names,
                    orientation='h',
                ),
                row=1, col=2
            )
        
        # 3. Learning Curves
        if "learning_curves" in data:
            curves = data["learning_curves"]
            
            for method_name, curve_data in curves.items():
                x = curve_data.get("x", [])
                y = curve_data.get("y", [])
                
                if x and y:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name=method_name
                        ),
                        row=2, col=1
                    )
        
        # 4. Resource Usage
        if "resource_usage" in data:
            resources = data["resource_usage"]
            
            if "timestamps" in resources:
                # Get relative timestamps
                timestamps = resources["timestamps"]
                start_time = timestamps[0]
                rel_times = [(t - start_time) for t in timestamps]
                
                # Plot CPU usage
                if "cpu_percent" in resources:
                    fig.add_trace(
                        go.Scatter(
                            x=rel_times,
                            y=resources["cpu_percent"],
                            name="CPU %",
                            line=dict(color="blue")
                        ),
                        row=2, col=2
                    )
                
                # Plot memory usage
                if "memory_percent" in resources:
                    fig.add_trace(
                        go.Scatter(
                            x=rel_times,
                            y=resources["memory_percent"],
                            name="Memory %",
                            line=dict(color="green")
                        ),
                        row=2, col=2
                    )
                
                # Plot GPU usage if available
                if "gpu_utilization" in resources and resources["gpu_utilization"][0] is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=rel_times,
                            y=resources["gpu_utilization"],
                            name="GPU %",
                            line=dict(color="magenta")
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        
        # Update x/y axis labels
        fig.update_xaxes(title_text="Trial", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Importance", row=1, col=2)
        
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="Reward", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
        fig.update_yaxes(title_text="Usage %", row=2, col=2)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, f"{filename}.html")
        fig.write_html(dashboard_path)
        logger.info(f"Saved HPO dashboard to {dashboard_path}")
        
        if self.show_plots:
            fig.show()
        
        return dashboard_path


# Test code
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create test visualizer
    viz = HPOVisualizer(output_dir="test_plots", interactive=True, static=True)
    
    # Test optimization history plot
    history_data = [
        {"trial_id": i, "value": (i/10) + ((i % 3) * 0.05)} for i in range(20)
    ]
    viz.plot_optimization_history(history_data, title="Test Optimization History")
    
    # Test parameter importance plot
    importance_data = {
        "learning_rate": 0.8,
        "batch_size": 0.5,
        "n_atoms": 0.3,
        "v_min": 0.2,
        "v_max": 0.15,
        "noisy_std": 0.1
    }
    viz.plot_parameter_importance(importance_data, title="Test Parameter Importance")
    
    # Test method comparison plot
    comparison_data = {
        "Bayesian": [0.85, 0.82, 0.9, 0.88, 0.83],
        "Evolutionary": [0.8, 0.75, 0.9, 0.82, 0.85],
        "Random Search": [0.7, 0.65, 0.8, 0.75, 0.72]
    }
    viz.plot_comparison(comparison_data, title="Test Method Comparison")
    
    # Test learning curves plot
    learning_data = {
        "Method1": {
            "x": list(range(100)),
            "y": [np.sin(i/5) * i/25 + 5 for i in range(100)],
            "std": [0.2 + 0.1 * np.sin(i/10) for i in range(100)]
        },
        "Method2": {
            "x": list(range(100)),
            "y": [np.cos(i/5) * i/20 + 4 + (i/100) for i in range(100)],
            "std": [0.3 + 0.05 * np.cos(i/10) for i in range(100)]
        }
    }
    viz.plot_learning_curves(learning_data, title="Test Learning Curves")
    
    # Combine all into a dashboard
    dashboard_data = {
        "optimization_history": {
            "trial_indices": [i for i in range(20)],
            "values": [(i/10) + ((i % 3) * 0.05) for i in range(20)],
            "best_so_far": [max((j/10) + ((j % 3) * 0.05) for j in range(i+1)) for i in range(20)]
        },
        "parameter_importance": importance_data,
        "learning_curves": learning_data,
        "resource_usage": {
            "timestamps": [time.time() + i for i in range(100)],
            "cpu_percent": [30 + 20 * np.sin(i/10) for i in range(100)],
            "memory_percent": [50 + 10 * np.cos(i/15) for i in range(100)],
            "gpu_utilization": [70 + 15 * np.sin(i/20) for i in range(100)]
        }
    }
    viz.generate_dashboard(dashboard_data, title="Test HPO Dashboard")