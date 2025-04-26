"""
Agent Builder module for Rainbow DQN HPO project.
Implements a customizable Rainbow DQN agent with SOTA features from DI-engine and Pearl libraries.
"""
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import logging
import os
from collections import deque
import random
import math
import copy
import time
from functools import partial
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/agent_builder.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration in Rainbow DQN (from the original Rainbow paper).
    Implements Factorized Gaussian noise for efficient exploration.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        # Sample noise if in training mode, otherwise use mu
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """
    Neural network for the Rainbow DQN algorithm.
    Combines features from DQN, Double DQN, Dueling DQN, and Distributional RL.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_noisy: bool = True,  # Flag to use noisy networks
    ):
        """
        Initialize the Rainbow network.
        
        Args:
            input_shape: Shape of the input observations (channels, height, width)
            action_dim: Number of possible actions
            n_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
            use_noisy: Whether to use NoisyNet for exploration
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.use_noisy = use_noisy
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the convolution output
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Choose between standard linear layers or noisy layers based on the flag
        if use_noisy:
            # Value stream with noisy layers (Dueling architecture)
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512),
                nn.ReLU(),
                NoisyLinear(512, n_atoms)
            )
            
            # Advantage stream with noisy layers (Dueling architecture)
            self.advantage_stream = nn.Sequential(
                NoisyLinear(conv_output_size, 512),
                nn.ReLU(),
                NoisyLinear(512, action_dim * n_atoms)
            )
        else:
            # Standard value stream (Dueling architecture)
            self.value_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_atoms)
            )
            
            # Standard advantage stream (Dueling architecture)
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim * n_atoms)
            )

    def _get_conv_output_size(self, input_shape: Tuple[int, ...]) -> int:
        """
        Calculate the size of the convolutional output.
        
        Args:
            input_shape: Shape of the input observations
            
        Returns:
            Size of the convolutional output
        """
        o = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(o.shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability distribution over atoms for each action
        """
        batch_size = x.size(0)
        
        # Get convolutional features
        conv_features = self.conv(x)
        conv_features = conv_features.view(batch_size, -1)
        
        # Dueling architecture
        values = self.value_stream(conv_features).view(batch_size, 1, self.n_atoms)
        advantages = self.advantage_stream(conv_features).view(batch_size, self.action_dim, self.n_atoms)
        
        # Combine value and advantage streams (Dueling)
        q_atoms = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities over atoms
        q_dist = torch.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """Reset noise for all noisy layers in the network"""
        if not self.use_noisy:
            return
            
        # Reset noise for all NoisyLinear layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for Rainbow DQN.
    Implements importance sampling and proportional prioritization.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize the PrioritizedReplayBuffer.
        
        Args:
            capacity: Maximum size of buffer
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
            beta_increment: Increment for beta parameter
            epsilon: Small positive value to avoid zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.position = 0
        self.size = 0
        
        # Initialize buffer
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        logger.info(f"Initialized PrioritizedReplayBuffer with capacity {capacity}, "
                   f"alpha={alpha}, beta={beta}")
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities = probabilities / probabilities.sum()
            
            indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** -self.beta
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities of sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation in replay buffer."""
    use_augmentation: bool = False 
    aug_type: str = "random_shift"  # Options: random_shift, cutout, random_conv, etc.
    aug_intensity: float = 0.1  # Intensity of augmentation


class DataAugmentation:
    """
    Data augmentation techniques for reinforcement learning.
    Inspired by Pearl and DI-engine approaches to improve sample efficiency.
    """
    
    @staticmethod
    def random_shift(states: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Apply random shift augmentation to batch of states.
        
        Args:
            states: Batch of states of shape (batch_size, channels, height, width)
            intensity: Maximum shift as a fraction of image size
            
        Returns:
            Augmented states
        """
        batch_size, channels, height, width = states.shape
        augmented = np.zeros_like(states)
        
        for i in range(batch_size):
            state = states[i]
            
            # Calculate maximum shifts
            max_h_shift = int(height * intensity)
            max_w_shift = int(width * intensity)
            
            # Random shifts
            h_shift = np.random.randint(-max_h_shift, max_h_shift + 1)
            w_shift = np.random.randint(-max_w_shift, max_w_shift + 1)
            
            # Apply shifts using roll (circular shift)
            if h_shift != 0:
                state = np.roll(state, h_shift, axis=1)
            if w_shift != 0:
                state = np.roll(state, w_shift, axis=2)
            
            augmented[i] = state
            
        return augmented
    
    @staticmethod
    def cutout(states: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Apply cutout augmentation to batch of states (masks random rectangles).
        
        Args:
            states: Batch of states of shape (batch_size, channels, height, width)
            intensity: Size of cutout as a fraction of image size
            
        Returns:
            Augmented states
        """
        batch_size, channels, height, width = states.shape
        augmented = states.copy()
        
        # Calculate cutout size
        size_h = max(1, int(height * intensity))
        size_w = max(1, int(width * intensity))
        
        for i in range(batch_size):
            # Random position
            top = np.random.randint(0, height - size_h + 1)
            left = np.random.randint(0, width - size_w + 1)
            
            # Apply cutout (set to zero)
            augmented[i, :, top:top+size_h, left:left+size_w] = 0
            
        return augmented
    
    @staticmethod
    def get_augmentation(aug_type: str) -> Callable:
        """
        Get augmentation function by name.
        
        Args:
            aug_type: Type of augmentation
            
        Returns:
            Augmentation function
        """
        augmentations = {
            "random_shift": DataAugmentation.random_shift,
            "cutout": DataAugmentation.cutout,
        }
        
        if aug_type not in augmentations:
            logger.warning(f"Unknown augmentation type: {aug_type}. Using random_shift.")
            return DataAugmentation.random_shift
        
        return augmentations[aug_type]


class AugmentedPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Prioritized Experience Replay buffer with data augmentation.
    Extends PrioritizedReplayBuffer with advanced data augmentation techniques.
    """
    
    def __init__(
        self,
        capacity: int,
        aug_config: AugmentationConfig,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize the AugmentedPrioritizedReplayBuffer.
        
        Args:
            capacity: Maximum size of buffer
            aug_config: Augmentation configuration
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
            beta_increment: Increment for beta parameter
            epsilon: Small positive value to avoid zero priority
        """
        super().__init__(capacity, alpha, beta, beta_increment, epsilon)
        self.aug_config = aug_config
        self.augmentation_fn = DataAugmentation.get_augmentation(aug_config.aug_type)
        
        logger.info(f"Initialized AugmentedPrioritizedReplayBuffer with augmentation: "
                    f"{aug_config.aug_type}, intensity: {aug_config.aug_intensity}")
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with optional data augmentation.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Call parent method to get the original sample
        states, actions, rewards, next_states, dones, indices, weights = super().sample(batch_size)
        
        # Apply data augmentation if enabled
        if self.aug_config.use_augmentation:
            states = self.augmentation_fn(states, self.aug_config.aug_intensity)
            next_states = self.augmentation_fn(next_states, self.aug_config.aug_intensity)
        
        return states, actions, rewards, next_states, dones, indices, weights


class RainbowDQNAgent:
    """
    Rainbow DQN agent implementing multiple improvements over vanilla DQN:
    - Double Q-learning
    - Dueling networks
    - Prioritized experience replay
    - Multi-step learning
    - Distributional RL
    - Noisy networks (partially implemented)
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        memory_size: int = 100000,
        n_steps: int = 3,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        seed: int = 42
    ):
        """
        Initialize the Rainbow DQN agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            learning_rate: Learning rate
            gamma: Discount factor
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            epsilon_start: Starting value of epsilon
            epsilon_end: Final value of epsilon
            epsilon_decay_steps: Number of steps to decay epsilon
            memory_size: Size of replay buffer
            n_steps: Number of steps for multi-step learning
            n_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
            seed: Random seed
        """
        self.action_space = action_space
        self.n_actions = action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.n_steps = n_steps
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.memory_size = memory_size
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        obs_shape = observation_space.shape
        
        self.policy_net = RainbowNetwork(obs_shape, self.n_actions, n_atoms, v_min, v_max).to(self.device)
        self.target_net = RainbowNetwork(obs_shape, self.n_actions, n_atoms, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # N-step learning buffer
        self.n_step_buffer = deque(maxlen=n_steps)
        
        # Training metrics
        self.training_steps = 0
        self.losses = []
        
        logger.info(f"Initialized Rainbow DQN agent with parameters:\n"
                   f"learning_rate={learning_rate}, gamma={gamma}, batch_size={batch_size}, "
                   f"target_update_freq={target_update_freq}, epsilon_start={epsilon_start}, "
                   f"epsilon_end={epsilon_end}, epsilon_decay_steps={epsilon_decay_steps}, "
                   f"memory_size={memory_size}, n_steps={n_steps}, n_atoms={n_atoms}, "
                   f"v_min={v_min}, v_max={v_max}, seed={seed}")
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eval_mode: Whether to use evaluation mode (no exploration)
            
        Returns:
            Selected action
        """
        # Use no exploration during evaluation
        if eval_mode:
            epsilon = 0.01
        else:
            epsilon = self.epsilon
        
        if random.random() > epsilon:
            # Convert state to tensor and get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action distribution
                q_dist = self.policy_net(state_tensor)
                
                # Calculate expected values using support
                q_values = torch.sum(q_dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
                
                # Select best action
                action = q_values.argmax(1).item()
        else:
            # Random action
            action = self.action_space.sample()
        
        # Decay epsilon
        if not eval_mode and self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        
        return action
    
    def _n_step_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """
        Calculate n-step return.
        
        Returns:
            Tuple of (state, action, n_step_reward, next_state, done)
        """
        R = sum([self.gamma**i * transition[2] for i, transition in enumerate(self.n_step_buffer)])
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        return state, action, R, next_state, done
    
    def store_transition(self, state: np.ndarray, action: int, 
                          reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store transition in replay buffer, handling n-step returns.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Store in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n_step_buffer is full or episode ended
        if len(self.n_step_buffer) == self.n_steps or done:
            # Calculate n-step return
            n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = self._n_step_return()
            
            # Store in replay buffer
            self.memory.add(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
            
            # Clear n-step buffer if episode ended
            if done:
                self.n_step_buffer.clear()
    
    def optimize(self) -> float:
        """
        Perform one step of optimization.
        
        Returns:
            Loss value
        """
        if self.memory.size < self.batch_size:
            return 0.0
        
        # Sample from memory
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Compute current Q distributions
        current_q_dist = self.policy_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        # Compute next Q distribution for DDQN
        with torch.no_grad():
            # Get next Q distributions (from policy network for action selection)
            next_q_dist = self.policy_net(next_states)
            next_actions = (next_q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2).argmax(dim=1)
            
            # Get target Q distributions (from target network for value estimation)
            target_q_dist = self.target_net(next_states)
            target_q_dist = target_q_dist[range(self.batch_size), next_actions]
            
            # Calculate target distribution (Bellman update)
            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_steps) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            
            # Compute L2 projection of Tz onto support z
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability of Tz to nearest atoms
            target_dist = torch.zeros_like(target_q_dist)
            
            for i in range(self.batch_size):
                for j in range(self.n_atoms):
                    l_idx = l[i, j]
                    u_idx = u[i, j]
                    
                    target_dist[i, l_idx] += target_q_dist[i, j] * (u_idx.float() - b[i, j])
                    target_dist[i, u_idx] += target_q_dist[i, j] * (b[i, j] - l_idx.float())
        
        # Cross-entropy loss with importance sampling weights
        loss = -(target_dist * torch.log(current_q_dist + 1e-8)).sum(dim=1)
        weighted_loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        new_priorities = loss.detach().cpu().numpy() + 1e-6  # Add a small constant to avoid zero priority
        self.memory.update_priorities(indices, new_priorities)
        
        # Update target network if needed
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        loss_value = weighted_loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path: str) -> None:
        """
        Save model to specified path.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from specified path.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"Model loaded from {path}")


class AgentBuilder:
    """Class for building Rainbow DQN agents with specified hyperparameters."""
    
    def __init__(self):
        """Initialize the AgentBuilder."""
        logger.info("AgentBuilder initialized")
    
    def build_agent(self, 
                   observation_space: gym.spaces.Box, 
                   action_space: gym.spaces.Discrete,
                   hyperparams: Dict[str, Any],
                   seed: int = 42) -> RainbowDQNAgent:
        """
        Build a Rainbow DQN agent with given hyperparameters.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            hyperparams: Dictionary of hyperparameters
            seed: Random seed
            
        Returns:
            Configured Rainbow DQN agent
        """
        default_params = {
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 32,
            "target_update_freq": 1000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 200000,
            "memory_size": 100000,
            "n_steps": 3,
            "n_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0
        }
        
        # Update default parameters with provided hyperparameters
        params = {**default_params, **hyperparams}
        
        logger.info(f"Building agent with parameters: {params}")
        
        return RainbowDQNAgent(
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            target_update_freq=params["target_update_freq"],
            epsilon_start=params["epsilon_start"],
            epsilon_end=params["epsilon_end"],
            epsilon_decay_steps=params["epsilon_decay_steps"],
            memory_size=params["memory_size"],
            n_steps=params["n_steps"],
            n_atoms=params["n_atoms"],
            v_min=params["v_min"],
            v_max=params["v_max"],
            seed=seed
        )

    def train_agent(self,
                  agent: RainbowDQNAgent,
                  env: gym.Env,
                  n_episodes: int,
                  max_steps_per_episode: int = 10000,
                  eval_interval: int = 10,
                  eval_episodes: int = 5,
                  save_path: str = None,
                  log_dir: str = "logs",
                  early_stopping_patience: int = 20,
                  early_stopping_threshold: float = None,
                  progress_callback: Callable[[Dict[str, Any]], None] = None) -> Dict[str, Any]:
        """
        Train a Rainbow DQN agent.
        
        Args:
            agent: Agent to train
            env: Environment to train in
            n_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            eval_interval: Interval to evaluate agent (in episodes)
            eval_episodes: Number of episodes for evaluation
            save_path: Path to save best model
            log_dir: Directory to save training logs
            early_stopping_patience: Number of evaluations with no improvement before stopping
            early_stopping_threshold: Reward threshold for early stopping
            progress_callback: Callback function for reporting progress
            
        Returns:
            Dictionary containing training history and best reward
        """
        os.makedirs(log_dir, exist_ok=True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "eval_rewards": [],
            "eval_steps": [],
            "losses": [],
            "timestamps": []
        }
        
        best_eval_reward = float('-inf')
        no_improvement_count = 0
        early_stopped = False
        start_time = time.time()
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_losses = []
            steps = 0
            done = False
            truncated = False
            
            # Single episode
            while not done and not truncated and steps < max_steps_per_episode:
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition and optimize
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.optimize()
                if loss > 0:
                    episode_losses.append(loss)
                
                # Move to the next state
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Record episode stats
            training_history["episode_rewards"].append(episode_reward)
            training_history["episode_lengths"].append(steps)
            training_history["timestamps"].append(time.time() - start_time)
            
            if episode_losses:
                training_history["losses"].append(np.mean(episode_losses))
            else:
                training_history["losses"].append(0.0)
            
            # Evaluation phase
            if (episode + 1) % eval_interval == 0:
                eval_rewards = []
                eval_lengths = []
                
                # Run evaluation episodes
                for _ in range(eval_episodes):
                    eval_state, _ = env.reset()
                    eval_episode_reward = 0
                    eval_steps = 0
                    eval_done = False
                    eval_truncated = False
                    
                    while not eval_done and not eval_truncated and eval_steps < max_steps_per_episode:
                        eval_action = agent.select_action(eval_state, eval_mode=True)
                        eval_state, eval_reward, eval_done, eval_truncated, _ = env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_steps += 1
                    
                    eval_rewards.append(eval_episode_reward)
                    eval_lengths.append(eval_steps)
                
                # Average evaluation metrics
                mean_eval_reward = np.mean(eval_rewards)
                training_history["eval_rewards"].append(mean_eval_reward)
                training_history["eval_steps"].append(np.mean(eval_lengths))
                
                # Log progress
                logger.info(f"Episode {episode+1}/{n_episodes}, "
                          f"Avg. Reward: {np.mean(training_history['episode_rewards'][-eval_interval:]):.2f}, "
                          f"Eval Reward: {mean_eval_reward:.2f}, "
                          f"Loss: {training_history['losses'][-1]:.6f}")
                
                # Save best model
                if mean_eval_reward > best_eval_reward:
                    best_eval_reward = mean_eval_reward
                    no_improvement_count = 0
                    if save_path:
                        agent.save(save_path)
                        logger.info(f"New best model saved with reward: {best_eval_reward:.2f}")
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} evaluations")
                    
                # Check for early stopping
                if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {episode+1} episodes")
                    early_stopped = True
                    break
                
                # Check for reward threshold early stopping
                if early_stopping_threshold is not None and mean_eval_reward >= early_stopping_threshold:
                    logger.info(f"Early stopping threshold reached: {mean_eval_reward:.2f} >= {early_stopping_threshold:.2f}")
                    early_stopped = True
                    break
                
                # Report progress if callback provided
                if progress_callback:
                    callback_data = {
                        "episode": episode + 1,
                        "total_episodes": n_episodes,
                        "eval_reward": mean_eval_reward,
                        "best_reward": best_eval_reward,
                        "no_improvement_count": no_improvement_count,
                        "early_stopped": early_stopped
                    }
                    progress_callback(callback_data)
        
        training_time = time.time() - start_time
        
        # Final results
        results = {
            "training_history": training_history,
            "best_eval_reward": best_eval_reward,
            "training_time": training_time,
            "early_stopped": early_stopped,
            "total_episodes": episode + 1
        }
        
        logger.info(f"Training completed. Best reward: {best_eval_reward:.2f}")
        logger.info(f"Total training time: {training_time:.2f}s")
        
        return results


if __name__ == "__main__":
    # Simple test to verify the agent builder works
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
    
    # Create fake environment spaces for testing
    obs_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
    act_space = Discrete(4)
    
    # Create builder
    builder = AgentBuilder()
    
    # Build agent with default hyperparameters
    agent = builder.build_agent(obs_space, act_space, {}, seed=42)
    
    # Test agent action selection
    fake_state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action(fake_state)
    
    print(f"Selected action: {action}")