from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

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


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        transitions = [self.buffer[idx] for idx in indices]
        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        rewards = [t.reward for t in transitions]
        next_states = [t.next_state for t in transitions]
        dones = [t.done for t in transitions]

        
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
