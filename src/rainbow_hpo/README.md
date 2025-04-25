# Rainbow DQN Hyperparameter Optimization Comparison

This project implements a comprehensive comparison between two hyperparameter optimization (HPO) methods for Rainbow DQN agents on Atari environments:

1. **Bayesian Optimization** (using Optuna)
2. **Evolutionary Algorithm** (using DEAP)

## Project Structure

The project is organized into the following modules:

- **env_builder.py**: Creates and configures Atari environments with appropriate wrappers
- **agent_builder.py**: Implements the Rainbow DQN agent with all its components
- **hpo_engine.py**: Contains the hyperparameter optimization engines (Bayesian and Evolutionary)
- **analyzer.py**: Provides tools to analyze and visualize the results of HPO methods
- **main.py**: Entry point that runs the comparison between the HPO methods

## Rainbow DQN Features

The Rainbow DQN implementation includes:

- Double Q-learning
- Dueling networks
- Prioritized experience replay
- Multi-step learning
- Distributional RL (C51)
- Noisy networks (partially implemented)

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
```

## Usage

### Basic Usage

Run the comparison with default parameters:

```bash
python main.py
```

This will run the comparison between Bayesian and Evolutionary optimization on the Breakout environment with 20 trials for each method.

### Command Line Arguments

- `--env`: Atari environment name (default: 'ALE/Breakout-v5')
- `--trials`: Number of trials for each HPO method (default: 20)
- `--seeds`: Number of seeds to evaluate each configuration (default: 3)
- `--base-seed`: Base random seed (default: 42)
- `--steps`: Number of training steps per trial (default: 500000)
- `--output-dir`: Directory to save results (default: 'models')
- `--list-games`: List available Atari games and exit

### Examples

Run with custom environment and fewer trials (for quicker results):

```bash
python main.py --env ALE/Pong-v5 --trials 10 --seeds 2 --steps 250000
```

List available Atari games:

```bash
python main.py --list-games
```

## Output

The comparison generates:

1. Trained model checkpoints
2. JSON files with optimization results
3. A comprehensive report with:
   - Comparison tables
   - Learning curves
   - Parameter importance visualizations
   - Reward distribution plots
   - Best hyperparameters for each method

## Hyperparameters

The following hyperparameters are tuned:

- `learning_rate`: Learning rate for the Adam optimizer
- `gamma`: Discount factor
- `batch_size`: Batch size for training
- `target_update_freq`: Frequency of target network updates
- `n_steps`: Number of steps for multi-step learning
- `epsilon_decay_steps`: Steps to decay exploration parameter

## License

This project is provided for educational purposes.

## References

- [Rainbow DQN Paper](https://arxiv.org/abs/1710.02298) - "Rainbow: Combining Improvements in Deep Reinforcement Learning"
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [DEAP Documentation](https://deap.readthedocs.io/)