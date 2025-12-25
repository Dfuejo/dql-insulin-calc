from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_range_metrics
from .replay_buffer import ReplayBuffer, Transition


def _device_or_default(device: Optional[str] = None) -> torch.device:
    """
    Choose device with preference: user-specified -> MPS -> CUDA -> CPU.
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
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    target_update_interval: int = 500
    target_tau: float = 1.0  # 1.0 => hard update, <1 => soft update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    hidden_sizes: Tuple[int, ...] = (128, 128)
    max_episodes: int = 500
    max_steps_per_episode: Optional[int] = None
    log_interval: int = 10
    device: Optional[str] = None
    gradient_clip_norm: Optional[float] = 1.0
    use_double_dqn: bool = True
    target_low: float = 70.0
    target_high: float = 180.0


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        self.config = config
        self.device = _device_or_default(config.device)
        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.total_steps = 0
        self.action_dim = action_dim

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, buffer: ReplayBuffer) -> Optional[float]:
        if len(buffer) < self.config.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute targets
        with torch.no_grad():
            if self.config.use_double_dqn:
                next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1).values
            targets = rewards + self.config.gamma * (1 - dones) * next_q

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.config.target_update_interval == 0:
            self._update_target_network()

        return float(loss.item())

    def _update_target_network(self) -> None:
        tau = self.config.target_tau
        if tau >= 1.0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)


def linear_epsilon(step: int, config: DQNConfig) -> float:
    fraction = min(1.0, step / max(1, config.epsilon_decay))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def train_dqn(env, config: DQNConfig) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    """
    Train a DQN agent in the provided environment.
    Returns the trained agent and a history dict.
    """
    state_dim = int(np.prod(env.reset().shape))
    action_dim = 2  # fixed for now: 0 or 1 (no insulin / insulin)
    agent = DQNAgent(state_dim, action_dim, config)
    buffer = ReplayBuffer(config.buffer_size)
    print(f"Training on device: {agent.device}")

    max_steps = config.max_steps_per_episode or env.params.max_steps
    history: Dict[str, List[float]] = {"episode_rewards": [], "losses": [], "tir": [], "tbr": [], "tor": []}

    global_step = 0
    for episode in range(1, config.max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        glucose_trace: List[float] = []

        for _ in range(max_steps):
            epsilon = linear_epsilon(global_step, config)
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

        history["episode_rewards"].append(episode_reward)
        metrics = compute_range_metrics(glucose_trace, target_low=config.target_low, target_high=config.target_high)
        history["tir"].append(metrics["tir"])
        history["tbr"].append(metrics["tbr"])
        history["tor"].append(metrics["tor"])

        if episode % config.log_interval == 0:
            avg_reward = np.mean(history["episode_rewards"][-config.log_interval :])
            avg_tir = np.mean(history["tir"][-config.log_interval :])
            avg_tbr = np.mean(history["tbr"][-config.log_interval :])
            avg_tor = np.mean(history["tor"][-config.log_interval :])
            print(
                f"Episode {episode}/{config.max_episodes} | "
                f"avg reward: {avg_reward:.3f} | "
                f"TIR/TBR/TOR: {avg_tir:.2f}/{avg_tbr:.2f}/{avg_tor:.2f} | "
                f"epsilon: {linear_epsilon(global_step, config):.3f} | "
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


__all__ = ["DQNConfig", "DQNAgent", "QNetwork", "train_dqn", "evaluate_policy"]
