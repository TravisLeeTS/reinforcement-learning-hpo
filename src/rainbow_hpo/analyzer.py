"""
Analyzer module for Rainbow DQN HPO project.
Provides visualization and analysis tools for hyperparameter tuning results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pickle
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/analyzer.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class HPOAnalysisResult:
    """Container for hyperparameter optimization analysis results."""
    best_params: Dict[str, Any]
    param_importance: Dict[str, float]
    performance_over_time: pd.DataFrame
    param_correlations: pd.DataFrame
    pca_components: Optional[np.ndarray] = None
    clustering_labels: Optional[np.ndarray] = None
    resource_usage_stats: Optional[Dict[str, Any]] = None


class HPOAnalyzer:
    """
    Analyzer for hyperparameter optimization results.
    Provides methods for visualization and analysis of HPO trials.
    """
    
    def __init__(self, results_path: str, output_dir: str = "analysis_results"):
        """
        Initialize the HPOAnalyzer.
        
        Args:
            results_path: Path to the HPO results file (JSON or pickle)
            output_dir: Directory to save analysis results
        """
        self.results_path = results_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        self.results = self._load_results(results_path)
        logger.info(f"Loaded {len(self.results)} trial results")
        
        # Convert to DataFrame for easier analysis
        self.df = self._create_dataframe()
        logger.info(f"Created analysis DataFrame with shape {self.df.shape}")
    
    def _load_results(self, path: str) -> List[Dict[str, Any]]:
        """
        Load results from file.
        
        Args:
            path: Path to results file
            
        Returns:
            List of trial results
        """
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.
        
        Returns:
            DataFrame with trial data
        """
        # Extract trial data
        trial_data = []
        
        for trial in self.results:
            # Create base entry with trial ID and value
            entry = {
                "trial_id": trial.get("trial_id", -1),
                "value": trial.get("value", float('nan')),
                "duration": trial.get("duration", float('nan')),
                "early_stopped": trial.get("early_stopped", False)
            }
            
            # Add parameters
            params = trial.get("params", {})
            for param_name, param_value in params.items():
                entry[f"param_{param_name}"] = param_value
            
            # Add resource usage
            resource_usage = trial.get("resource_usage", {})
            for metric, metric_value in resource_usage.items():
                entry[f"resource_{metric}"] = metric_value
            
            # Add metadata
            metadata = trial.get("metadata", {})
            for meta_key, meta_value in metadata.items():
                if isinstance(meta_value, (int, float, str, bool)):
                    entry[f"meta_{meta_key}"] = meta_value
            
            trial_data.append(entry)
        
        return pd.DataFrame(trial_data)
    
    def analyze(self) -> HPOAnalysisResult:
        """
        Perform comprehensive analysis of HPO results.
        
        Returns:
            HPOAnalysisResult object with analysis results
        """
        # Find best trial
        best_idx = self.df["value"].idxmax()
        best_params = {}
        for col in self.df.columns:
            if col.startswith("param_"):
                param_name = col[6:]  # Remove "param_" prefix
                best_params[param_name] = self.df.loc[best_idx, col]
        
        # Calculate parameter importance (correlation with value)
        param_importance = {}
        param_cols = [col for col in self.df.columns if col.startswith("param_")]
        
        for col in param_cols:
            param_name = col[6:]  # Remove "param_" prefix
            
            # Check data type for appropriate correlation method
            if pd.api.types.is_numeric_dtype(self.df[col]):
                corr, _ = stats.spearmanr(self.df[col], self.df["value"], nan_policy="omit")
                if np.isnan(corr):
                    corr = 0.0
            else:
                # For categorical parameters, use ANOVA
                categories = self.df[col].unique()
                if len(categories) > 1:
                    groups = [self.df[self.df[col] == cat]["value"].dropna() for cat in categories]
                    try:
                        f_val, p_val = stats.f_oneway(*groups)
                        corr = f_val / (1 + f_val)  # Normalize to 0-1 scale
                    except:
                        corr = 0.0
                else:
                    corr = 0.0
            
            param_importance[param_name] = abs(corr)
        
        # Sort by importance
        param_importance = {k: v for k, v in sorted(param_importance.items(), key=lambda x: x[1], reverse=True)}
        
        # Performance over time
        performance_df = self.df.sort_values("trial_id")[["trial_id", "value", "duration", "early_stopped"]]
        performance_df["cumulative_best"] = performance_df["value"].cummax()
        performance_df["cumulative_time"] = performance_df["duration"].cumsum()
        
        # Parameter correlations
        numeric_param_cols = [col for col in param_cols if pd.api.types.is_numeric_dtype(self.df[col])]
        if numeric_param_cols:
            corr_df = self.df[numeric_param_cols].corr(method="spearman")
        else:
            corr_df = pd.DataFrame()
        
        # Resource usage statistics
        resource_cols = [col for col in self.df.columns if col.startswith("resource_")]
        resource_stats = {}
        if resource_cols:
            for col in resource_cols:
                metric = col[9:]  # Remove "resource_" prefix
                resource_stats[metric] = {
                    "mean": self.df[col].mean(),
                    "std": self.df[col].std(),
                    "min": self.df[col].min(),
                    "max": self.df[col].max()
                }
        
        # Create and return analysis result
        return HPOAnalysisResult(
            best_params=best_params,
            param_importance=param_importance,
            performance_over_time=performance_df,
            param_correlations=corr_df,
            resource_usage_stats=resource_stats
        )
    
    def visualize(self, analysis_result: Optional[HPOAnalysisResult] = None) -> None:
        """
        Generate visualization plots for HPO results.
        
        Args:
            analysis_result: Pre-computed analysis result, or None to compute now
        """
        if analysis_result is None:
            analysis_result = self.analyze()
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Performance over time
        self._plot_performance_over_time(analysis_result.performance_over_time, plots_dir)
        
        # 2. Parameter importance
        self._plot_parameter_importance(analysis_result.param_importance, plots_dir)
        
        # 3. Parameter correlations
        if not analysis_result.param_correlations.empty:
            self._plot_parameter_correlations(analysis_result.param_correlations, plots_dir)
        
        # 4. Parameter distributions
        self._plot_parameter_distributions(plots_dir)
        
        # 5. Parameter pairwise relationships
        self._plot_parameter_pairplot(plots_dir)
        
        # 6. Interactive parallel coordinates plot
        self._plot_parallel_coordinates(plots_dir)
        
        # 7. Resource usage
        if analysis_result.resource_usage_stats:
            self._plot_resource_usage(plots_dir)
        
        logger.info(f"Visualization plots saved to {plots_dir}")
    
    def _plot_performance_over_time(self, performance_df: pd.DataFrame, plots_dir: str) -> None:
        """Plot performance metrics over time."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(performance_df["trial_id"], performance_df["value"], "o-", alpha=0.5, label="Trial Value")
        plt.plot(performance_df["trial_id"], performance_df["cumulative_best"], "r-", label="Best Value")
        plt.xlabel("Trial ID")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Performance Over Trials")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(performance_df["cumulative_time"], performance_df["cumulative_best"], "g-")
        plt.xlabel("Cumulative Time (seconds)")
        plt.ylabel("Best Value")
        plt.title("Best Value vs. Cumulative Time")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_over_time.png"), dpi=300)
        plt.close()
        
        # Interactive version with plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(x=performance_df["trial_id"], y=performance_df["value"], 
                      mode="markers+lines", name="Trial Value", opacity=0.6),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=performance_df["trial_id"], y=performance_df["cumulative_best"], 
                      mode="lines", name="Best Value", line=dict(color="red", width=2)),
            row=1, col=1
        )
        
        # Add early stopped markers
        if performance_df["early_stopped"].any():
            early_stopped = performance_df[performance_df["early_stopped"]]
            fig.add_trace(
                go.Scatter(x=early_stopped["trial_id"], y=early_stopped["value"],
                          mode="markers", name="Early Stopped", 
                          marker=dict(color="red", symbol="x", size=10)),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=performance_df["cumulative_time"], y=performance_df["cumulative_best"],
                      mode="lines", name="Best vs Time", line=dict(color="green", width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Performance Over Trials",
            height=800,
            xaxis2_title="Cumulative Time (seconds)",
            yaxis_title="Value",
            yaxis2_title="Best Value"
        )
        
        fig.write_html(os.path.join(plots_dir, "performance_over_time.html"))
    
    def _plot_parameter_importance(self, param_importance: Dict[str, float], plots_dir: str) -> None:
        """Plot parameter importance."""
        plt.figure(figsize=(10, 6))
        
        # Convert to series for easier plotting
        importance_series = pd.Series(param_importance).sort_values(ascending=True)
        
        importance_series.plot(kind="barh")
        plt.xlabel("Importance")
        plt.ylabel("Parameter")
        plt.title("Parameter Importance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "parameter_importance.png"), dpi=300)
        plt.close()
        
        # Interactive version with plotly
        fig = px.bar(
            x=list(importance_series.values),
            y=list(importance_series.index),
            orientation="h",
            labels={"x": "Importance", "y": "Parameter"},
            title="Parameter Importance"
        )
        
        fig.update_layout(height=600)
        fig.write_html(os.path.join(plots_dir, "parameter_importance.html"))
    
    def _plot_parameter_correlations(self, corr_df: pd.DataFrame, plots_dir: str) -> None:
        """Plot parameter correlations heatmap."""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(corr_df, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
        plt.title("Parameter Correlations")
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "parameter_correlations.png"), dpi=300)
        plt.close()
        
        # Interactive version with plotly
        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            origin="lower",
            labels=dict(color="Correlation"),
            title="Parameter Correlations"
        )
        
        fig.update_layout(height=800, width=800)
        fig.write_html(os.path.join(plots_dir, "parameter_correlations.html"))
    
    def _plot_parameter_distributions(self, plots_dir: str) -> None:
        """Plot parameter distributions with value coloring."""
        param_cols = [col for col in self.df.columns if col.startswith("param_")]
        
        for i, col in enumerate(param_cols):
            param_name = col[6:]  # Remove "param_" prefix
            
            plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # For numeric parameters, create a scatter plot
                plt.scatter(self.df[col], self.df["value"], alpha=0.7)
                plt.xlabel(param_name)
                plt.ylabel("Value")
                plt.title(f"Value vs. {param_name}")
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                try:
                    z = np.polyfit(self.df[col], self.df["value"], 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(self.df[col]), p(sorted(self.df[col])), "r--", alpha=0.7)
                except:
                    pass
            else:
                # For categorical parameters, create a box plot
                sns.boxplot(x=col, y="value", data=self.df)
                plt.xlabel(param_name)
                plt.ylabel("Value")
                plt.title(f"Value Distribution by {param_name}")
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"param_{param_name}_distribution.png"), dpi=300)
            plt.close()
            
            # Interactive version with plotly
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fig = px.scatter(
                    self.df,
                    x=col,
                    y="value",
                    trendline="ols",
                    labels={col: param_name, "value": "Value"},
                    title=f"Value vs. {param_name}"
                )
            else:
                fig = px.box(
                    self.df,
                    x=col,
                    y="value",
                    labels={col: param_name, "value": "Value"},
                    title=f"Value Distribution by {param_name}"
                )
            
            fig.write_html(os.path.join(plots_dir, f"param_{param_name}_distribution.html"))
    
    def _plot_parameter_pairplot(self, plots_dir: str) -> None:
        """Plot pairwise relationships between parameters."""
        # Select numeric parameters and limit to most important ones to avoid too many plots
        numeric_param_cols = [col for col in self.df.columns if col.startswith("param_") and pd.api.types.is_numeric_dtype(self.df[col])]
        
        if len(numeric_param_cols) < 2:
            logger.info("Not enough numeric parameters for pairplot")
            return
        
        # Limit to top 6 parameters if there are too many
        if len(numeric_param_cols) > 6:
            numeric_param_cols = numeric_param_cols[:6]
        
        # Add value column
        plot_df = self.df[numeric_param_cols + ["value"]].copy()
        
        # Rename columns for better readability
        rename_dict = {col: col[6:] for col in numeric_param_cols}
        plot_df = plot_df.rename(columns=rename_dict)
        
        plt.figure(figsize=(12, 10))
        sns.pairplot(plot_df, hue="value", palette="viridis", diag_kind="kde", corner=True)
        plt.suptitle("Parameter Pairwise Relationships", y=1.02)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "parameter_pairplot.png"), dpi=300)
        plt.close()
    
    def _plot_parallel_coordinates(self, plots_dir: str) -> None:
        """Create interactive parallel coordinates plot."""
        # Get parameter columns
        param_cols = [col for col in self.df.columns if col.startswith("param_")]
        
        if not param_cols:
            return
        
        # Prepare data
        plot_df = self.df[param_cols + ["value", "trial_id"]].copy()
        
        # Rename columns for better readability
        rename_dict = {col: col[6:] for col in param_cols}
        plot_df = plot_df.rename(columns=rename_dict)
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            plot_df,
            color="value",
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parameter Relationships (Parallel Coordinates)",
            labels={col: col for col in plot_df.columns}
        )
        
        fig.update_layout(
            font=dict(size=10),
            height=600,
            coloraxis_colorbar=dict(title="Value")
        )
        
        fig.write_html(os.path.join(plots_dir, "parallel_coordinates.html"))
    
    def _plot_resource_usage(self, plots_dir: str) -> None:
        """Plot resource usage metrics."""
        # Get resource columns
        resource_cols = [col for col in self.df.columns if col.startswith("resource_")]
        
        if not resource_cols:
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for i, col in enumerate(resource_cols):
            metric = col[9:]  # Remove "resource_" prefix
            
            plt.subplot(len(resource_cols), 1, i+1)
            plt.plot(self.df["trial_id"], self.df[col], "o-", alpha=0.7)
            plt.xlabel("Trial ID" if i == len(resource_cols)-1 else "")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
        
        plt.suptitle("Resource Usage Over Trials")
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "resource_usage.png"), dpi=300)
        plt.close()
        
        # Interactive version with plotly
        fig = make_subplots(rows=len(resource_cols), cols=1, shared_xaxes=True)
        
        for i, col in enumerate(resource_cols):
            metric = col[9:]  # Remove "resource_" prefix
            
            fig.add_trace(
                go.Scatter(x=self.df["trial_id"], y=self.df[col], mode="lines+markers", name=metric),
                row=i+1, col=1
            )
            
            fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        fig.update_layout(
            height=200 * len(resource_cols),
            title_text="Resource Usage Over Trials",
            showlegend=True,
            xaxis_title="Trial ID"
        )
        
        fig.write_html(os.path.join(plots_dir, "resource_usage.html"))
    
    def export_results(self, analysis_result: Optional[HPOAnalysisResult] = None) -> None:
        """
        Export analysis results to various formats.
        
        Args:
            analysis_result: Pre-computed analysis result, or None to compute now
        """
        if analysis_result is None:
            analysis_result = self.analyze()
        
        # Export to JSON
        export_data = {
            "best_params": analysis_result.best_params,
            "param_importance": analysis_result.param_importance,
            "performance_over_time": analysis_result.performance_over_time.to_dict(orient="records"),
            "resource_usage_stats": analysis_result.resource_usage_stats
        }
        
        json_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=4)
        
        # Export to CSV
        csv_path = os.path.join(self.output_dir, "trial_data.csv")
        self.df.to_csv(csv_path, index=False)
        
        logger.info(f"Analysis results exported to {self.output_dir}")


if __name__ == "__main__":
    # Simple test to verify the analyzer works
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "checkpoints/trial_results.json"
    
    analyzer = HPOAnalyzer(results_path, output_dir="analysis_results")
    analysis_result = analyzer.analyze()
    analyzer.visualize(analysis_result)
    analyzer.export_results(analysis_result)
    
    print(f"Best parameters: {analysis_result.best_params}")
    print(f"Parameter importance: {analysis_result.param_importance}")