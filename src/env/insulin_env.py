"""
Simplified insulin/glucose control environment.

State: (G, dG, CH) where
    G  = current blood glucose (mg/dL)
    dG = delta glucose since last step (mg/dL)
    CH = active carbohydrates still being absorbed (grams)

Action space: [0, 1]
    0 -> do nothing
    1 -> deliver a fixed bolus of rapid-acting insulin

The dynamics are intentionally lightweight so the environment can run
on a normal laptop while remaining configurable for more realistic
simulations later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EnvParams:

    """Configurable hyperparameters for the environment."""

    target_glucose: float = 110.0
    min_glucose: float = 40.0
    max_glucose: float = 400.0

    meal_probability: float = 0.05  # probability of a meal each step
    meal_size_range: Tuple[float, float] = (10.0, 80.0)  # grams

    insulin_units_per_action: float = 1.0  # bolus size for action=1
    insulin_sensitivity: float = 5.0  # mg/dL drop per unit insulin per step
    insulin_decay: float = 0.05  # fraction of insulin cleared each step

    carb_absorption: float = 0.05  # fraction of active carbs absorbed per step
    basal_drift: float = 0.1  # natural upward drift per step
    noise_std: float = 3.0  # stochasticity in glucose dynamics

    time_step_min: int = 5
    max_steps: int = 288  # roughly a day at 5-minute steps
    initial_glucose_range: Tuple[float, float] = (90.0, 140.0)

    # Normalization constants used for NN inputs
    glucose_scale: float = 300.0
    delta_glucose_scale: float = 30.0
    carb_scale: float = 120.0


class InsulinEnv:
    """
    Minimal environment resembling OpenAI Gym's interface (reset/step).
    """

    def __init__(self, params: EnvParams | None = None, seed: int | None = None):
        self.params = params or EnvParams()
        self.rng = np.random.default_rng(seed)
        self.state: np.ndarray | None = None
        self.prev_glucose: float = 0.0
        self.active_carbs: float = 0.0
        self.active_insulin: float = 0.0
        self.steps: int = 0

    def reset(self, glucose: float | None = None) -> np.ndarray:
        """Reset environment to an initial state."""
        if glucose is None:
            glucose = float(
                self.rng.uniform(
                    self.params.initial_glucose_range[0],
                    self.params.initial_glucose_range[1],
                )
            )
        self.prev_glucose = glucose
        self.active_carbs = 0.0
        self.active_insulin = 0.0
        self.steps = 0
        self.state = self._build_state(glucose, 0.0, self.active_carbs)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action and return (state, reward, done, info).
        Action must be 0 or 1.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1.")

        p = self.params
        self.steps += 1

        # Sample meal event
        meal = 0.0
        if self.rng.random() < p.meal_probability:
            meal = float(self.rng.uniform(*p.meal_size_range))
        self.active_carbs += meal

        # Apply insulin bolus
        if action == 1:
            self.active_insulin += p.insulin_units_per_action

        # Effects
        carb_effect = p.carb_absorption * self.active_carbs
        insulin_effect = p.insulin_sensitivity * self.active_insulin
        noise = float(self.rng.normal(0.0, p.noise_std))

        delta_glucose = carb_effect - insulin_effect + p.basal_drift + noise
        current_glucose = self.prev_glucose + delta_glucose
        self.prev_glucose = current_glucose

        # Update pools (simple exponential decay)
        self.active_carbs = max(0.0, self.active_carbs - carb_effect)
        self.active_insulin = max(0.0, self.active_insulin - p.insulin_decay * self.active_insulin)

        reward = self._compute_reward(current_glucose, action)
        done = (
            current_glucose <= p.min_glucose
            or current_glucose >= p.max_glucose
            or self.steps >= p.max_steps
        )

        self.state = self._build_state(current_glucose, delta_glucose, self.active_carbs)
        info = {"glucose": current_glucose, "meal": meal, "active_carbs": self.active_carbs}
        return self.state.copy(), reward, done, info

    def _compute_reward(self, glucose: float, action: int) -> float:
        """
        Reward encourages staying near target_glucose, with extra penalty for insulin use
        and for hypoglycemia/hyperglycemia.
        """
        p = self.params
        deviation = abs(glucose - p.target_glucose)
        reward = -deviation / 50.0  # scaled deviation penalty

        if glucose < 70:
            reward -= 4.0  # strong penalty for hypo
        elif glucose > 180:
            reward -= 1.0  # penalty for sustained highs

        reward -= 0.1 * action  # discourage unnecessary boluses
        return reward

    def _build_state(self, glucose: float, delta_glucose: float, carbs: float) -> np.ndarray:
        """Normalize features for NN consumption."""
        p = self.params
        normalized = np.array(
            [
                glucose / p.glucose_scale,
                delta_glucose / p.delta_glucose_scale,
                carbs / p.carb_scale,
            ],
            dtype=np.float32,
        )
        return normalized


__all__ = ["EnvParams", "InsulinEnv"]
