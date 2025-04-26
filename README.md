# RL Hyperparameter Optimization Project

This project implements advanced hyperparameter optimization (HPO) for reinforcement learning algorithms, with a special focus on Rainbow DQN for both Atari games and continuous control environments.

## Features

- **Multiple HPO Techniques**:
  - Bayesian Optimization with Optuna
  - Evolutionary Algorithms with DEAP
  - Population-Based Training with Ray Tune
  - Grid Search for baseline comparison
  
- **Rainbow DQN Implementation** with all components:
  - Double Q-learning
  - Dueling networks
  - Prioritized experience replay (PER)
  - Multi-step learning
  - Distributional RL (C51)
  - Noisy networks
  
- **Advanced HPO Engine**:
  - Automatic early stopping for inefficient trials
  - Checkpointing for resilience against crashes
  - Resource monitoring to prevent OOM errors
  - Parallelization across trials for speed
  - Parameter importance analysis
  
- **Comprehensive Analysis**:
  - Statistical comparison across multiple methods
  - Parameter importance visualization
  - Learning curves and reward distributions
  - Detailed performance metrics

- **Environment Support**:
  - Atari games (via Gymnasium)
  - Classic control environments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl_hpo.git
cd rl_hpo

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running HPO on a continuous control task:
```bash
python src/hpo_pendulum.py --optimize --n-trials 10
```

### Training with the best hyperparameters found:
```bash
python src/hpo_pendulum.py --train --render
```

### For Atari environments:
```bash
python src/rainbow_hpo/atari_hpo_experiment.py --env PongNoFrameskip-v4 --bayesian
```

## Project Structure

- `src/rainbow_hpo/`: Main Rainbow DQN implementation and HPO framework
- `src/models/`: Saved models from optimization runs
- `src/logs/`: Training logs and evaluation results

For detailed documentation on the Rainbow HPO framework, see [Rainbow HPO README](src/rainbow_hpo/README.md).

## GitHub Integration

This project includes built-in GitHub integration. To push your project to GitHub:

1. **Automatic Method**:
   - Run the main script `src/hpo_pendulum.py`
   - At the end of execution, you will be prompted whether you want to push to GitHub
   - Enter 'yes' when prompted
   - Provide your GitHub username and desired repository name
   - The script will handle repository creation and pushing automatically

2. **Manual Method**:
   - From a Python script:
   ```python
   from src.hpo_pendulum import push_to_github
   
   # Replace with your info
   push_to_github(repo_name="rl-hyperparameter-optimization", username="your-username")
   ```

3. **Command Line Method**:
   ```bash
   # Initialize Git repository
   git init
   
   # Add all files
   git add .
   
   # Commit changes
   git commit -m "Initial commit: RL HPO project"
   
   # Add remote repository (create repository on GitHub first)
   git remote add origin https://github.com/your-username/your-repo-name.git
   
   # Push to GitHub
   git push -u origin master
   ```

The project includes a comprehensive `.gitignore` file to exclude unnecessary files from version control.

## License

This project is provided for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```
@software{rl_hpo2025,
  author = {Your Name},
  title = {Reinforcement Learning Hyperparameter Optimization},
  year = {2025},
  url = {https://github.com/yourusername/rl_hpo}
}
```