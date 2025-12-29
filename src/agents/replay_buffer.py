from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch

"""Simple experience replay buffer for DQN agent.
Stores transitions and allows sampling random batches.
Transition = (state, action, reward, next_state, done)

The Neural Network waits until enough samples are collected ( min_buffer_size ) before training begins
Then the agent samples batches of transitions from the buffer to update the Q-network to break correlation between consecutive samples.
"""

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    n_step: int = 1


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[Tuple[torch.Tensor, ...], None, None]:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        transitions = [self.buffer[idx] for idx in indices]
        return self._to_tensors(transitions), None, None

    @staticmethod
    def _to_tensors(transitions: List[Transition]) -> Tuple[torch.Tensor, ...]:
        states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in transitions], dtype=torch.int64)
        rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32)
        dones = torch.as_tensor([t.done for t in transitions], dtype=torch.float32)
        n_steps = torch.as_tensor([t.n_step for t in transitions], dtype=torch.float32)
        return states, actions, rewards, next_states, dones, n_steps

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Simple proportional PER with TD-error priorities."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-3):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priorities: Deque[float] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_prio)

    def sample(self, batch_size: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor | None, np.ndarray]:
        if len(self.buffer) == len(self.priorities) == 0:
            raise ValueError("Cannot sample from empty buffer")
        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        transitions = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        batch = self._to_tensors(transitions)
        return batch, torch.as_tensor(weights, dtype=torch.float32), indices

    def update_priorities(self, indices: List[int], td_errors: torch.Tensor) -> None:
        td = td_errors.detach().cpu().numpy()
        for idx, err in zip(indices, td):
            prio = float(abs(err) + self.eps)
            if idx < len(self.priorities):
                self.priorities[idx] = prio
