# Rainbow DQN Hyperparameter Optimization Framework

This project implements a comprehensive comparison between different hyperparameter optimization (HPO) methods for Rainbow DQN agents on both Atari environments and continuous control tasks.

## HPO Methods Supported

1. **Bayesian Optimization** (using Optuna)
2. **Evolutionary Algorithm** (using DEAP)
3. **Population-Based Training** (using Ray Tune)
4. **Grid Search** (as a baseline comparison)

## New Features (April 2025 Update)

- **Enhanced Optimization Engine**:
  - Automatic early stopping for inefficient trials
  - Checkpointing and resume capability
  - System resource monitoring to prevent OOM errors
  - Improved parallelization across trials
  - Trial-level isolation for more robust experimentation
  
- **Visualization Enhancements**:
  - Interactive learning curves with Plotly
  - Parameter importance analysis
  - Run comparison across multiple environments

## Project Structure

The project is organized into the following modules:

- **env_builder.py**: Creates and configures environments with appropriate wrappers
- **agent_builder.py**: Implements the Rainbow DQN agent with all its components
- **hpo_engine.py**: Contains the hyperparameter optimization engines with the new enhancements
- **analyzer.py**: Provides tools to analyze and visualize the results of HPO methods
- **main.py**: Entry point for standard Rainbow DQN experiments
- **hpo_comparison.py**: Specialized tool for comparing multiple HPO methods in parallel
- **atari_hpo_experiment.py**: Optimized for running HPO specifically on Atari environments

## Rainbow DQN Features

The Rainbow DQN implementation includes:

- Double Q-learning
- Dueling networks
- Prioritized experience replay
- Multi-step learning
- Distributional RL (C51)
- Noisy networks

## Requirements

```
stable-baselines3
optuna
gymnasium[atari]
gymnasium[accept-rom-license]
torch
matplotlib
numpy
plotly
deap
tqdm
typing-extensions
ale-py
autorom[accept-rom-license]
opencv-python
psutil
tensorboard
pandas
seaborn
ray[tune]
```

## Quick Start Guide

### Basic Usage

Run a simple HPO experiment:

```bash
python main.py --optimize --env Pendulum-v1 --n-trials 10
```

Train an agent with the best hyperparameters found:

```bash
python main.py --train --env Pendulum-v1 --render
```

### With Optimization Enhancements

Run HPO with parallel trials and early stopping:

```bash
python main.py --optimize --env Pendulum-v1 --n-trials 20 --parallel 4
```

Resume an interrupted optimization:

```bash
python main.py --optimize --env Pendulum-v1 --resume
```

### Advanced HPO Comparison

For a comprehensive comparison of all HPO methods:

```bash
python hpo_comparison.py --env ALE/Breakout-v5 --trials 20 --seeds 3
```

To run only a specific HPO method:

```bash
python hpo_comparison.py --env ALE/Pong-v5 --bayesian
```

### Atari-Specific Experiments

For optimizing Rainbow DQN on Atari environments:

```bash
python atari_hpo_experiment.py --env PongNoFrameskip-v4 --all --train-best
```

## Detailed Command Line Arguments

### For main.py
- `--optimize`: Run hyperparameter optimization
- `--train`: Train with the best found hyperparameters
- `--n-trials`: Number of HPO trials (default: 10)
- `--env`: Environment ID (default: 'Pendulum-v1')
- `--seed`: Random seed (default: 42)
- `--output-dir`: Directory to save results (default: 'models')
- `--render`: Render environment when training
- `--list-games`: List available Atari games and exit
- `--parallel`: Number of parallel trials to run (default: 1)
- `--early-stopping`: Enable early stopping for unpromising trials (default: True)
- `--resume`: Resume optimization from a checkpoint
- `--monitor-resources`: Monitor system resources during optimization (default: True)
- `--checkpoint-interval`: Interval for saving checkpoints (default: 5 trials)

### For hpo_comparison.py
- `--env`: Gym environment ID
- `--trials`: Number of trials per HPO method
- `--seeds`: Number of random seeds for statistical reliability
- `--budget`: Training budget per trial in environment steps
- `--seed`: Master random seed
- `--output-dir`: Directory to save comparison results
- `--bayesian`: Run only Bayesian Optimization
- `--evolutionary`: Run only Evolutionary Algorithm
- `--pbt`: Run only Population-Based Training
- `--grid`: Run only Grid Search
- `--parallel`: Number of parallel trials (default: 1)
- `--resume`: Resume optimization from checkpoint

### For atari_hpo_experiment.py
- `--env`: Atari environment ID
- `--trials`: Number of trials per HPO method
- `--seeds`: Number of seeds to run for statistical reliability
- `--budget`: Training budget per trial in steps
- `--seed`: Master random seed
- `--output-dir`: Directory to save results
- `--all`: Run all HPO strategies
- `--bayesian`: Run Bayesian Optimization
- `--evolutionary`: Run Evolutionary Algorithm
- `--pbt`: Run Population-Based Training
- `--train-best`: Train an agent using the best found hyperparameters
- `--render`: Render the environment when training the best agent
- `--parallel`: Number of parallel trials (default: 1)
- `--checkpoint-interval`: Interval for saving checkpoints (default: 5 trials)

## Advanced Usage Examples

### Complete HPO comparison with visualization

```bash
python hpo_comparison.py --env ALE/Breakout-v5 --trials 30 --seeds 5 --budget 1000000 --parallel 4
```

### Quick experiment on Pendulum with resource monitoring

```bash
python main.py --optimize --env Pendulum-v1 --n-trials 10 --monitor-resources
```

### Train with best parameters and visualize performance

```bash
python main.py --train --env Pendulum-v1 --render --analyze
```

### Resuming an interrupted Atari experiment

```bash
python atari_hpo_experiment.py --env PongNoFrameskip-v4 --bayesian --resume
```

### Running PBT with custom checkpoint interval

```bash
python hpo_comparison.py --env ALE/Pong-v5 --pbt --parallel 8 --checkpoint-interval 2
```

## Understanding the Optimization Engine

The HPO engine (`hpo_engine.py`) now includes several key optimizations:

### Early Stopping

Unpromising trials are automatically terminated early to save computational resources:
- For Bayesian Optimization: Uses Optuna's pruning mechanisms
- For Evolutionary Algorithms: Terminates low-performing individuals
- For Population-Based Training: Replaces poorly-performing models

### Checkpointing System

The optimizers periodically save their state, allowing you to:
- Resume from interruptions or system crashes
- Monitor progress in real-time
- Analyze partial results before completion

### Resource Monitoring

The system monitors CPU and memory usage to:
- Prevent out-of-memory (OOM) errors during training
- Automatically scale back resource usage when needed
- Provide usage statistics in the final report

### Parallelization

Trials can be run in parallel, dramatically speeding up the optimization process:
- Uses Python's multiprocessing for CPU-bound workloads
- Integrates with Ray for more sophisticated parallelism (PBT)
- Automatically manages resources across trials

## Environment Wrappers

The project includes several custom environment wrappers implemented in `env_builder.py`:

### ActionDiscretizationWrapper

Enables discrete action algorithms like Rainbow DQN to work with continuous action space environments.

### RewardScalingWrapper

Scales rewards by a constant factor, which helps stabilize training in environments with very large or small rewards.

### ObservationNormalizationWrapper

Normalizes observations to have zero mean and unit variance using running statistics.

### EpisodeLengthLimiter

Limits the maximum episode length, useful for environments with no natural termination.

## Output Files

The optimization process generates:

1. **Checkpoints**: Saved in the output directory under `checkpoints/`
   - `hpo_checkpoint.pkl`: Binary checkpoint file for resuming
   - `hpo_status.json`: Human-readable status of optimization
   - `param_importances.json`: Parameter importance statistics

2. **Models**: Saved in the output directory under `models/`
   - `trial_X/best_model.zip`: Best model found in each trial
   - `best_overall.zip`: Best model across all trials

3. **Results**: Saved in the output directory
   - `hpo_results.json`: Complete results of optimization
   - `comparison.csv`: Tabular data for HPO method comparison
   - `performance_metrics.json`: Detailed metrics for each trial

4. **Visualizations**: Generated in the output directory under `plots/`
   - Learning curves
   - Parameter importance charts
   - Method comparison bar plots
   - Resource utilization graphs

## License

This project is provided for educational and research purposes.

## References

- [Rainbow DQN Paper](https://arxiv.org/abs/1710.02298) - "Rainbow: Combining Improvements in Deep Reinforcement Learning"
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)