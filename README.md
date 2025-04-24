# Reinforcement Learning Hyperparameter Optimization

This project implements hyperparameter optimization (HPO) for reinforcement learning algorithms on various environments from OpenAI Gym/Gymnasium.

## Features

- Hyperparameter optimization using multiple techniques:
  - Bayesian Optimization with Optuna
  - Evolutionary Algorithms with DEAP
- Support for multiple RL algorithms:
  - Proximal Policy Optimization (PPO)
  - Advantage Actor-Critic (A2C)
  - Rainbow DQN (with Double Q-learning, Dueling networks, PER, etc.)
- Training pause feature when reward threshold is reached
- Automatic model evaluation and saving
- Visualization of optimization results
- GitHub integration for easy sharing

## Requirements

All dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

## Usage

### Pendulum HPO

To run the hyperparameter optimization for Pendulum:

```bash
python src/hpo_pendulum.py
```

The script will:
1. Run multiple trials to optimize hyperparameters
2. Save the best model and optimization results
3. Train a final model with the best hyperparameters
4. Offer to push the project to GitHub

### Rainbow DQN HPO Comparison

For comparing HPO methods on Rainbow DQN with Atari environments:

```bash
python src/rainbow_hpo/main.py
```

This will run a comparison between Bayesian and Evolutionary optimization on the Breakout environment with 20 trials for each method.

Command line arguments:
- `--env`: Atari environment name (default: 'ALE/Breakout-v5')
- `--trials`: Number of trials for each HPO method (default: 20)
- `--seeds`: Number of seeds to evaluate each configuration (default: 3)
- `--steps`: Number of training steps per trial (default: 500000)

Example for quicker results:
```bash
python src/rainbow_hpo/main.py --env ALE/Pong-v5 --trials 10 --seeds 2 --steps 250000
```

## Project Structure

```
rl_hpo/
│
├── requirements.txt   # Dependencies
├── README.md          # This file
│
├── src/               # Source code
│   ├── hpo_pendulum.py  # Pendulum HPO implementation
│   ├── logs/          # Training logs for Pendulum
│   ├── models/        # Saved models from trials
│   │
│   └── rainbow_hpo/   # Rainbow DQN HPO comparison
│       ├── agent_builder.py  # Rainbow DQN implementation
│       ├── analyzer.py       # Results visualization
│       ├── env_builder.py    # Environment creation
│       ├── hpo_engine.py     # HPO methods implementation
│       ├── main.py           # Entry point
│       ├── README.md         # Rainbow-specific documentation
│       ├── configs/          # Configuration files
│       ├── logs/             # Training logs
│       └── models/           # Saved models
│
└── docs/              # Documentation and reports
```

## Rainbow DQN Implementation

The Rainbow DQN implementation includes:
- Double Q-learning
- Dueling networks
- Prioritized experience replay
- Multi-step learning
- Distributional RL (C51)
- Noisy networks (partial)

## Customization

The code includes callbacks that pause training when reward thresholds are reached. You can modify these thresholds in the respective scripts.

To run HPO on different environments, modify the environment creation functions in the respective modules.

## License

MIT