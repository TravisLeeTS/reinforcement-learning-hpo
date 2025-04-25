"""
Environment Builder module for Rainbow DQN HPO project.
Implements wrappers and environment setup logic with inspiration from DI-engine.
"""

import gymnasium as gym
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from gym.wrappers import AtariPreprocessing, FrameStack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/env_builder.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ActionDiscretizationWrapper(gym.ActionWrapper):
    """
    Discretizes continuous action spaces for use with discrete action algorithms.
    Useful for environments like Pendulum where we want to use Rainbow DQN.
    """
    
    def __init__(self, env: gym.Env, n_bins: int = 11):
        """
        Initialize the ActionDiscretizationWrapper.
        
        Args:
            env: Gym environment with continuous action space
            n_bins: Number of bins for discretization
        """
        super().__init__(env)
        
        assert isinstance(env.action_space, gym.spaces.Box), "This wrapper only works with continuous action spaces"
        
        self.n_bins = n_bins
        self.action_dim = env.action_space.shape[0]
        self.low = env.action_space.low
        self.high = env.action_space.high
        
        # Define the new discrete action space
        self.action_space = gym.spaces.Discrete(n_bins ** self.action_dim)
        
        logger.info(f"Discretized action space with {n_bins} bins per dimension")
        logger.info(f"Original space: {env.action_space}, New space: {self.action_space}")
    
    def action(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous action.
        
        Args:
            action: Discrete action
            
        Returns:
            Corresponding continuous action
        """
        # Convert action index to a tuple of indices for each dimension
        indices = []
        remaining = action
        
        for _ in range(self.action_dim):
            indices.append(remaining % self.n_bins)
            remaining //= self.n_bins
        
        # Convert indices to continuous values
        continuous_action = []
        
        for idx, (low, high) in zip(indices, zip(self.low, self.high)):
            # Map from [0, n_bins-1] to [low, high]
            ratio = idx / (self.n_bins - 1)
            value = low + ratio * (high - low)
            continuous_action.append(value)
        
        return np.array(continuous_action, dtype=np.float32)


class RewardScalingWrapper(gym.RewardWrapper):
    """
    Scales rewards by a constant factor.
    Useful for environments with very large or small rewards.
    """
    
    def __init__(self, env: gym.Env, scale: float = 0.1):
        """
        Initialize the RewardScalingWrapper.
        
        Args:
            env: Gym environment
            scale: Scaling factor for rewards
        """
        super().__init__(env)
        self.scale = scale
        logger.info(f"Scaling rewards by factor {scale}")
    
    def reward(self, reward: float) -> float:
        """
        Scale the reward.
        
        Args:
            reward: Original reward
            
        Returns:
            Scaled reward
        """
        return self.scale * reward


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """
    Normalizes observations to have zero mean and unit variance.
    Uses running statistics for online normalization.
    """
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """
        Initialize the ObservationNormalizationWrapper.
        
        Args:
            env: Gym environment
            epsilon: Small constant to avoid division by zero
        """
        super().__init__(env)
        self.epsilon = epsilon
        
        # Initialize running statistics
        self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.running_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 0
        
        logger.info(f"Initializing observation normalization for shape {env.observation_space.shape}")
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize the observation.
        
        Args:
            observation: Original observation
            
        Returns:
            Normalized observation
        """
        # Update running statistics
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        delta2 = observation - self.running_mean
        self.running_var += delta * delta2
        
        # Calculate standard deviation
        if self.count > 1:
            std = np.sqrt(self.running_var / self.count)
        else:
            std = np.ones_like(self.running_var)
        
        # Normalize
        normalized_obs = (observation - self.running_mean) / (std + self.epsilon)
        
        return normalized_obs


class EpisodeLengthLimiter(gym.Wrapper):
    """
    Limits the maximum length of an episode.
    Useful for environments with no natural termination.
    """
    
    def __init__(self, env: gym.Env, max_steps: int = 1000):
        """
        Initialize the EpisodeLengthLimiter.
        
        Args:
            env: Gym environment
            max_steps: Maximum steps per episode
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0
        
        logger.info(f"Limiting episode length to {max_steps} steps")
    
    def reset(self, **kwargs):
        """Reset the environment and step counter."""
        self.current_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Take a step and check if max length is reached."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info


class EnvironmentBuilder:
    """Class for building and configuring environments for Rainbow DQN."""
    
    def __init__(self):
        """Initialize the EnvironmentBuilder."""
        logger.info("EnvironmentBuilder initialized")
    
    def _apply_atari_wrappers(self, env: gym.Env) -> gym.Env:
        """
        Apply standard Atari preprocessing wrappers.
        
        Args:
            env: Gym environment
            
        Returns:
            Wrapped environment
        """
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False
        )
        env = FrameStack(env, 4)
        
        logger.info("Applied Atari preprocessing wrappers")
        return env
    
    def build_env(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        use_action_discretization: bool = True,
        use_reward_scaling: bool = True,
        use_obs_normalization: bool = True,
        max_episode_steps: int = 1000,
        discretization_bins: int = 11,
        reward_scale: float = 0.1
    ) -> gym.Env:
        """
        Build and configure a Gym environment for Rainbow DQN.
        
        Args:
            env_id: Gym environment ID (e.g., 'Pendulum-v1')
            render_mode: Rendering mode (None, 'human', 'rgb_array')
            use_action_discretization: Whether to discretize continuous action spaces
            use_reward_scaling: Whether to scale rewards
            use_obs_normalization: Whether to normalize observations
            max_episode_steps: Maximum steps per episode
            discretization_bins: Number of bins for action discretization
            reward_scale: Scaling factor for rewards
            
        Returns:
            Configured Gym environment
        """
        # Create base environment
        env = gym.make(env_id, render_mode=render_mode)
        
        logger.info(f"Created environment: {env_id}")
        
        # Apply wrappers based on environment type
        if "Atari" in env_id or env_id.startswith("ALE/"):
            env = self._apply_atari_wrappers(env)
        
        # Apply general wrappers
        if use_action_discretization and isinstance(env.action_space, gym.spaces.Box):
            env = ActionDiscretizationWrapper(env, n_bins=discretization_bins)
        
        if use_reward_scaling:
            env = RewardScalingWrapper(env, scale=reward_scale)
        
        if use_obs_normalization:
            env = ObservationNormalizationWrapper(env)
        
        # Limit episode length
        env = EpisodeLengthLimiter(env, max_steps=max_episode_steps)
        
        return env