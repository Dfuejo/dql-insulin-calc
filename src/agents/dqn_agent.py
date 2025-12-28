from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .metrics import compute_range_metrics
from .replay_buffer import ReplayBuffer, Transition


def _device_or_default(device: Optional[str] = None) -> torch.device:
    """
    Choose device with preference: user-specified -> MPS -> CUDA -> CPU.
    My machine is MPS-capable, so I want to test that path
    """
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Feed-forward network producing Q-values for each action."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class DQNConfig:

    # Q-learning hyperparameters
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_eta: float = 30_000.0  # exponential decay scale

    # Replay buffer parameters
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    
    # Target network update parameters
    target_update_interval: int = 500
    target_tau: float = 1.0  # 1.0 => hard update, <1 => soft update
    hidden_sizes: Tuple[int, ...] = (256, 256)
    max_episodes: int = 500
    max_steps_per_episode: Optional[int] = None
    log_interval: int = 10 # log every n episodes
    device: Optional[str] = None
    gradient_clip_norm: Optional[float] = 1.0
    target_low: float = 70.0
    target_high: float = 180.0

    # Double DQN flag 
    """ The diference between DQN and Double DQN is in the way target Q-values are computed during training.
        In Double DQN, the action selection and action evaluation are decoupled to reduce overestimation bias.
        First policy selects the best action for the next state, then target network evaluates that action to get the Q-value"""
    use_double_dqn: bool = True
    

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        
        self.config = config
        self.device = _device_or_default(config.device)
        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # optimizer adam ( adaptive moment estimation )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.total_steps = 0
        self.action_dim = action_dim

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Convert state to tensor and add batch dimension (from shape (state_dim,) to (1, state_dim))
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get Q-values from policy network (torch.no_grad() no gradient needed )
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, buffer: ReplayBuffer) -> Optional[float]:
        """Sample a batch from the replay buffer and perform a DQN update step."""
        if len(buffer) < self.config.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.config.batch_size)
        # Move/copy to device ( where the model is located )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute targets
        with torch.no_grad():

            # Double DQN target calculation
            if self.config.use_double_dqn:
                next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1).values
            targets = rewards + self.config.gamma * (1 - dones) * next_q

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss function is Mean Squared Error between current Q-values and targets
        loss = F.mse_loss(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping 
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        # Update target network periodically 
        self.total_steps += 1
        if self.total_steps % self.config.target_update_interval == 0:
            self._update_target_network()

        return float(loss.item())


    def _update_target_network(self) -> None:
        """Update target network parameters (depending on tau)"""
        tau = self.config.target_tau
        if tau >= 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)


def exp_epsilon(step: int, config: DQNConfig) -> float:
    """
    Exponential epsilon schedule:
        eps(t) = eps_f + (eps_0 - eps_f) * exp(-t / eta)
    """
    return float(
        config.epsilon_end
        + (config.epsilon_start - config.epsilon_end) * math.exp(-step / max(1.0, config.epsilon_eta))
    )


def train_dqn(env, config: DQNConfig) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    """
    Train a DQN agent in the provided environment.
    Returns the trained agent and a history dict.
    """
    state_dim = int(np.prod(env.reset().shape))
    action_dim = 2  # fixed for now 0 or 1 (no insulin / insulin); could generalize to more realistic action scope
    agent = DQNAgent(state_dim, action_dim, config)
    buffer = ReplayBuffer(config.buffer_size)
    print(f"Training on device: {agent.device}")

    max_steps = config.max_steps_per_episode or env.params.max_steps
    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "losses": [],
        "tir": [],
        "tbr": [],
        "tor": [],
        "interval_stats": [],
    }

    global_step = 0

    for episode in range(1, config.max_episodes + 1):

        state = env.reset()
        episode_reward = 0.0
        glucose_trace: List[float] = []

        for _ in range(max_steps):

            epsilon = exp_epsilon(global_step, config)
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            buffer.push(transition)

            loss = agent.update(buffer)
            if loss is not None:
                history["losses"].append(loss)

            state = next_state
            episode_reward += reward
            global_step += 1
            glucose_trace.append(float(info.get("glucose", 0.0)))
            if done:
                break
        
        # End of episode, log metrics

        history["episode_rewards"].append(episode_reward)
        metrics = compute_range_metrics(glucose_trace, target_low=config.target_low, target_high=config.target_high)
        history["tir"].append(metrics["tir"])
        history["tbr"].append(metrics["tbr"])
        history["tor"].append(metrics["tor"])

        # For logging after every log_interval episodes giving average metrics (now is every 10 episodes)
        if episode % config.log_interval == 0:
            avg_reward = np.mean(history["episode_rewards"][-config.log_interval :])
            avg_tir = np.mean(history["tir"][-config.log_interval :])
            avg_tbr = np.mean(history["tbr"][-config.log_interval :])
            avg_tor = np.mean(history["tor"][-config.log_interval :])
            var_reward = np.var(history["episode_rewards"][-config.log_interval :]) if len(history["episode_rewards"]) >= config.log_interval else 0.0
            var_tir = np.var(history["tir"][-config.log_interval :]) if len(history["tir"]) >= config.log_interval else 0.0
            history["interval_stats"].append(
                {
                    "episode": episode,
                    "mean_reward": float(avg_reward),
                    "var_reward": float(var_reward),
                    "mean_tir": float(avg_tir),
                    "var_tir": float(var_tir),
                    "mean_tbr": float(avg_tbr),
                    "mean_tor": float(avg_tor),
                    "epsilon": float(exp_epsilon(global_step, config)),
                }
            )
            print(
                f"Episode {episode}/{config.max_episodes} | "
                f"avg reward: {avg_reward:.3f} | "
                f"TIR/TBR/TOR: {avg_tir:.2f}/{avg_tbr:.2f}/{avg_tor:.2f} | "
                f"epsilon: {exp_epsilon(global_step, config):.3f} | "
                f"buffer: {len(buffer)}"
            )

    return agent, history


def evaluate_policy(
    env,
    agent: DQNAgent,
    episodes: int = 5,
    max_steps: Optional[int] = None,
    target_low: float = 70.0,
    target_high: float = 180.0,
) -> Dict[str, object]:
    """
    Evaluate a trained policy (epsilon=0) and report TIR/TBR/TOR and rewards.
    """
    prev_mode = agent.policy_net.training
    agent.policy_net.eval()

    max_steps = max_steps or env.params.max_steps
    episode_metrics: List[Dict[str, float]] = []
    episode_rewards: List[float] = []
    episode_traces: List[List[float]] = []

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            glucose_trace: List[float] = []
            total_reward = 0.0

            for _ in range(max_steps):
                action = agent.act(state, epsilon=0.0)
                next_state, reward, done, info = env.step(action)
                glucose_trace.append(float(info.get("glucose", 0.0)))
                total_reward += reward
                state = next_state
                if done:
                    break

            metrics = compute_range_metrics(glucose_trace, target_low=target_low, target_high=target_high)
            episode_metrics.append(metrics)
            episode_rewards.append(total_reward)
            episode_traces.append(glucose_trace)

    if prev_mode:
        agent.policy_net.train()

    mean_metrics = {
        "tir": float(np.mean([m["tir"] for m in episode_metrics])) if episode_metrics else 0.0,
        "tbr": float(np.mean([m["tbr"] for m in episode_metrics])) if episode_metrics else 0.0,
        "tor": float(np.mean([m["tor"] for m in episode_metrics])) if episode_metrics else 0.0,
    }

    return {
        "episode_rewards": episode_rewards,
        "episode_metrics": episode_metrics,
        "mean_metrics": mean_metrics,
        "glucose_traces": episode_traces,
    }


def save_policy(agent: DQNAgent, path: str) -> None:
    torch.save(agent.policy_net.state_dict(), path)


def load_policy(agent: DQNAgent, path: str) -> None:
    state = torch.load(path, map_location=agent.device)
    agent.policy_net.load_state_dict(state)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())


__all__ = [
    "DQNConfig",
    "DQNAgent",
    "QNetwork",
    "train_dqn",
    "evaluate_policy",
    "save_policy",
    "load_policy",
]
