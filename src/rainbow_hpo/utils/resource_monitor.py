"""
Resource monitoring utilities for HPO experiments.
Tracks system resources (CPU, memory, GPU) during optimization.
"""

import os
import time
import threading
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Resource monitoring dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. CPU and memory monitoring will be limited.")

# GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

if not TORCH_AVAILABLE and not GPUTIL_AVAILABLE:
    logger.warning("Neither torch nor GPUtil available. GPU monitoring disabled.")


class ResourceMonitor:
    """
    Monitors system resources (CPU, memory, GPU) during optimization.
    Runs in a background thread and records usage statistics.
    """
    
    def __init__(self, monitor_gpu: bool = True,
                interval: float = 1.0,
                log_to_file: bool = True,
                log_dir: str = "resource_logs",
                verbose: bool = False,
                warning_threshold: float = 0.85):
        """
        Initialize the resource monitor.
        
        Args:
            monitor_gpu: Whether to monitor GPU usage if available
            interval: Time between measurements in seconds
            log_to_file: Whether to save logs to file
            log_dir: Directory for resource usage logs
            verbose: Whether to print resource usage to console
            warning_threshold: Fraction of resource usage to trigger warnings
        """
        self.monitor_gpu = monitor_gpu and (TORCH_AVAILABLE or GPUTIL_AVAILABLE)
        self.interval = interval
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.verbose = verbose
        self.warning_threshold = warning_threshold
        
        # Initialize storage for metrics
        self.cpu_percent = []
        self.memory_percent = []
        self.memory_used = []
        self.memory_available = []
        self.gpu_utilization = []
        self.gpu_memory_used = []
        self.timestamps = []
        
        self._running = False
        self._thread = None
        
        # Create log directory if needed
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Set up unique log file name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = os.path.join(self.log_dir, f"resource_usage_{timestamp}.csv")
            
            # Write header to log file
            with open(self.log_filename, 'w') as f:
                header = "timestamp,cpu_percent,memory_percent,memory_used_mb,memory_available_mb"
                if self.monitor_gpu:
                    header += ",gpu_utilization,gpu_memory_used_mb"
                f.write(header + "\n")
        
        # Check if we have the needed packages
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. Using basic resource monitoring.")
    
    def start(self):
        """Start resource monitoring in a background thread."""
        if self._running:
            logger.warning("Resource monitor already running.")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        if self.verbose:
            logger.info("Resource monitoring started.")
    
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            if self._thread.is_alive():
                self._thread.join(timeout=2*self.interval)
            self._thread = None
        
        if self.verbose:
            logger.info("Resource monitoring stopped.")
        
        # Save summary to log file
        if self.log_to_file:
            self._save_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a background thread."""
        while self._running:
            try:
                current_time = time.time()
                self.timestamps.append(current_time)
                
                # Get CPU usage
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    memory_used = memory.used / (1024 * 1024)  # MB
                    memory_available = memory.available / (1024 * 1024)  # MB
                else:
                    # Fallback to minimal info from os module
                    import os
                    cpu_percent = 0.0
                    try:
                        memory_used = 0.0
                        memory_available = 0.0
                        memory_percent = 0.0
                    except:
                        memory_used = 0.0
                        memory_available = 0.0
                        memory_percent = 0.0
                
                self.cpu_percent.append(cpu_percent)
                self.memory_percent.append(memory_percent)
                self.memory_used.append(memory_used)
                self.memory_available.append(memory_available)
                
                # Get GPU usage if available and enabled
                gpu_utilization = 0.0
                gpu_memory_used = 0.0
                
                if self.monitor_gpu:
                    if GPUTIL_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                # Use first GPU for now
                                gpu = gpus[0]
                                gpu_utilization = gpu.load * 100.0
                                gpu_memory_used = gpu.memoryUsed
                        except Exception as e:
                            logger.debug(f"Error getting GPU metrics: {e}")
                    
                    elif TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            # Get memory usage from torch
                            gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
                            gpu_utilization = 0.0  # torch doesn't provide utilization
                        except Exception as e:
                            logger.debug(f"Error getting GPU metrics from torch: {e}")
                
                self.gpu_utilization.append(gpu_utilization)
                self.gpu_memory_used.append(gpu_memory_used)
                
                # Check for high usage warnings
                self._check_warnings(cpu_percent, memory_percent, gpu_utilization)
                
                # Log to file if enabled
                if self.log_to_file:
                    with open(self.log_filename, 'a') as f:
                        line = f"{current_time},{cpu_percent},{memory_percent},{memory_used},{memory_available}"
                        if self.monitor_gpu:
                            line += f",{gpu_utilization},{gpu_memory_used}"
                        f.write(line + "\n")
                
                # Print current usage if verbose
                if self.verbose and len(self.timestamps) % 10 == 0:
                    logger.info(f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}% " + 
                              (f"GPU: {gpu_utilization:.1f}%, GPU Memory: {gpu_memory_used:.0f}MB" 
                               if self.monitor_gpu else ""))
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            # Sleep until next measurement
            time.sleep(self.interval)
    
    def _check_warnings(self, cpu_percent, memory_percent, gpu_utilization):
        """Check for high resource usage and log warnings."""
        if cpu_percent > self.warning_threshold * 100:
            logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
        
        if memory_percent > self.warning_threshold * 100:
            logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        
        if self.monitor_gpu and gpu_utilization > self.warning_threshold * 100:
            logger.warning(f"High GPU usage detected: {gpu_utilization:.1f}%")
    
    def _save_summary(self):
        """Save a summary of resource usage statistics."""
        if not self.timestamps:
            return  # No data collected
        
        # Calculate statistics
        stats = {
            "cpu_percent": self._calculate_stats(self.cpu_percent),
            "memory_percent": self._calculate_stats(self.memory_percent),
            "memory_used_mb": self._calculate_stats(self.memory_used),
            "memory_available_mb": self._calculate_stats(self.memory_available),
        }
        
        if self.monitor_gpu:
            stats.update({
                "gpu_utilization": self._calculate_stats(self.gpu_utilization),
                "gpu_memory_used_mb": self._calculate_stats(self.gpu_memory_used),
            })
        
        # Save summary to file
        if self.log_to_file:
            summary_file = os.path.join(self.log_dir, "resource_summary.csv")
            first_write = not os.path.exists(summary_file)
            
            with open(summary_file, 'a') as f:
                if first_write:
                    header = "timestamp,duration_sec,cpu_mean,cpu_max,memory_mean,memory_max"
                    if self.monitor_gpu:
                        header += ",gpu_mean,gpu_max,gpu_memory_mean,gpu_memory_max"
                    f.write(header + "\n")
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
                
                line = (f"{timestamp},{duration:.1f}," +
                       f"{stats['cpu_percent']['mean']:.1f},{stats['cpu_percent']['max']:.1f}," +
                       f"{stats['memory_percent']['mean']:.1f},{stats['memory_percent']['max']:.1f}")
                
                if self.monitor_gpu:
                    line += (f",{stats['gpu_utilization']['mean']:.1f},{stats['gpu_utilization']['max']:.1f}," +
                           f"{stats['gpu_memory_used_mb']['mean']:.1f},{stats['gpu_memory_used_mb']['max']:.1f}")
                
                f.write(line + "\n")
        
        return stats
    
    def _calculate_stats(self, values):
        """Calculate statistics for a list of values."""
        if not values:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        
        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "max": float(np.max(values_array)),
            "min": float(np.min(values_array)),
            "std": float(np.std(values_array)),
            "last": float(values_array[-1]) if len(values_array) > 0 else 0.0
        }
    
    def get_usage_stats(self):
        """
        Get summary statistics of resource usage.
        
        Returns:
            Dictionary with resource usage statistics
        """
        return {
            "cpu_percent": self._calculate_stats(self.cpu_percent),
            "memory_percent": self._calculate_stats(self.memory_percent),
            "memory_used_mb": self._calculate_stats(self.memory_used),
            "memory_available_mb": self._calculate_stats(self.memory_available),
            "gpu_utilization": self._calculate_stats(self.gpu_utilization) if self.monitor_gpu else None,
            "gpu_memory_used_mb": self._calculate_stats(self.gpu_memory_used) if self.monitor_gpu else None,
            "duration_sec": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0,
            "n_measurements": len(self.timestamps)
        }
    
    def get_current_usage(self):
        """
        Get the most recent resource usage measurements.
        
        Returns:
            Dictionary with current resource usage
        """
        if not self.timestamps:
            return None
            
        return {
            "timestamp": self.timestamps[-1] if self.timestamps else time.time(),
            "cpu_percent": self.cpu_percent[-1] if self.cpu_percent else 0.0,
            "memory_percent": self.memory_percent[-1] if self.memory_percent else 0.0,
            "memory_used_mb": self.memory_used[-1] if self.memory_used else 0.0,
            "memory_available_mb": self.memory_available[-1] if self.memory_available else 0.0,
            "gpu_utilization": self.gpu_utilization[-1] if self.gpu_utilization and self.monitor_gpu else 0.0,
            "gpu_memory_used_mb": self.gpu_memory_used[-1] if self.gpu_memory_used and self.monitor_gpu else 0.0,
        }
    
    def plot_resource_usage(self, output_file=None):
        """
        Plot resource usage over time.
        
        Args:
            output_file: File path to save the plot (None for display only)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.timestamps:
                logger.warning("No resource data to plot")
                return
            
            # Convert timestamps to relative seconds
            start_time = self.timestamps[0]
            rel_times = [(t - start_time) for t in self.timestamps]
            
            # Create figure with two Y axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot CPU and memory on first axis
            ax1.plot(rel_times, self.cpu_percent, 'b-', label='CPU %')
            ax1.plot(rel_times, self.memory_percent, 'g-', label='Memory %')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Usage %')
            ax1.set_ylim(0, 100)
            
            # Plot memory usage on second axis
            memory_gb = [m/1024 for m in self.memory_used]  # Convert to GB
            ax2.plot(rel_times, memory_gb, 'r-', label='Memory Used (GB)')
            ax2.set_ylabel('Memory (GB)')
            
            # Add GPU if monitored
            if self.monitor_gpu:
                ax1.plot(rel_times, self.gpu_utilization, 'm-', label='GPU %')
                gpu_memory_gb = [m/1024 for m in self.gpu_memory_used]  # Convert to GB
                ax2.plot(rel_times, gpu_memory_gb, 'c-', label='GPU Memory (GB)')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid and title
            ax1.grid(True, alpha=0.3)
            plt.title('System Resource Usage During Optimization')
            plt.tight_layout()
            
            # Save or show
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Resource usage plot saved to {output_file}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            logger.error("matplotlib is required for plotting resource usage")
        except Exception as e:
            logger.error(f"Error plotting resource usage: {e}")


# Test code
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create monitor
    monitor = ResourceMonitor(verbose=True, interval=0.5)
    
    try:
        # Start monitoring
        logger.info("Starting resource monitoring for 10 seconds...")
        monitor.start()
        
        # Create some load
        for i in range(5):
            logger.info(f"Creating some CPU load... ({i+1}/5)")
            # CPU load
            _ = [i*i for i in range(10000000)]
            # Memory load
            big_list = [0] * 1000000
            time.sleep(1)
        
        # Get current usage
        usage = monitor.get_current_usage()
        logger.info(f"Current usage: CPU {usage['cpu_percent']:.1f}%, Memory {usage['memory_percent']:.1f}%")
        
        # Get overall stats
        stats = monitor.get_usage_stats()
        logger.info(f"Average CPU: {stats['cpu_percent']['mean']:.1f}%, Max: {stats['cpu_percent']['max']:.1f}%")
        logger.info(f"Average Memory: {stats['memory_percent']['mean']:.1f}%, Max: {stats['memory_percent']['max']:.1f}%")
        
        # Plot
        monitor.plot_resource_usage("resource_test_plot.png")
        
    finally:
        # Stop monitoring
        monitor.stop()
        logger.info("Resource monitoring stopped.")