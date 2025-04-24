# Reinforcement Learning Hyperparameter Optimization

This project implements hyperparameter optimization (HPO) for reinforcement learning algorithms on the CartPole-v1 environment from OpenAI Gym/Gymnasium.

## Features

- Hyperparameter optimization using Optuna
- Support for multiple RL algorithms:
  - Proximal Policy Optimization (PPO)
  - Advantage Actor-Critic (A2C)
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

To run the hyperparameter optimization:

```bash
python src/hpo_pendulum.py
```

The script will:
1. Run multiple trials to optimize hyperparameters
2. Save the best model and optimization results
3. Train a final model with the best hyperparameters
4. Offer to push the project to GitHub

## Project Structure

```
rl_hpo/
│
├── requirements.txt   # Dependencies
├── README.md          # This file
│
├── src/               # Source code
│   └── hpo_pendulum.py  # Main script with HPO implementation
│
├── logs/              # Training logs and metrics
│   ├── evaluations.npz
│   ├── monitor.csv
│   ├── optimization_history.png
│   └── param_importance.png
│
└── models/            # Saved models from trials and final training
    └── best_model.zip
```

## Customization

The code includes a `PauseCallback` class that pauses training when a certain reward threshold is reached. You can modify this threshold by changing the `reward_threshold` parameter.

To run HPO on a different environment, modify the `create_env()` function.

## License

MIT