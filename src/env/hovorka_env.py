"""
Discrete-time Hovorka model environment.

Implements a simplified Hovorka model for insulin-glucose dynamics with
subcutaneous insulin absorption, gut absorption, and insulin action compartments.
The API matches InsulinEnv: reset() -> state, step(action) -> (state, reward, done, info).

State: (G, dG, CH) normalized similarly to InsulinEnv.
Action: 0 (no bolus) or 1 (deliver fixed bolus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class HovorkaParams:
    # Model parameters (defaults from typical adult; tune as needed)
    BW: float = 70.0  # body weight (kg)
    VG: float = 0.16  # glucose distribution volume (L/kg)
    VI: float = 0.12  # insulin distribution volume (L/kg)
    ke: float = 0.138  # 1/min insulin elimination
    ka1: float = 0.006  # 1/min rate constants for insulin action
    ka2: float = 0.06
    ka3: float = 0.03
    F01: float = 0.0097  # mmol/kg/min non-insulin glucose flux
    k12: float = 0.066  # 1/min transfer from non-accessible to accessible compartment
    EGP0: float = 0.0161  # mmol/kg/min endogenous glucose production
    taumin: float = 20.0  # min minimum absorption time

    # Gut absorption
    AG: float = 0.8  # carbohydrate bioavailability
    tau_g: float = 40.0  # min gastric emptying
    Dmax: float = 100.0  # g max CHO per meal (for limiting absorption)

    # Subcutaneous insulin absorption
    kabs: float = 0.002  # 1/min absorption
    kt: float = 0.02  # 1/min degradation

    # Simulation settings
    dt: float = 5.0  # minutes per step
    max_steps: int = 288  # 24h at 5-min steps
    target_glucose: float = 110.0  # mg/dL
    min_glucose: float = 40.0
    max_glucose: float = 400.0
    noise_std: float = 2.0  # mg/dL measurement noise

    # Bolus dosing
    insulin_action_levels: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)  # U bolus options

    # Normalization for state output
    glucose_scale: float = 300.0
    delta_glucose_scale: float = 30.0
    carb_scale: float = 120.0


class HovorkaEnv:
    def __init__(self, params: HovorkaParams | None = None, seed: int | None = None):
        self.params = params or HovorkaParams()
        self.rng = np.random.default_rng(seed)
        self.action_dim = len(self.params.insulin_action_levels)
        self.reset()

    def reset(self, glucose: float | None = None) -> np.ndarray:
        p = self.params
        if glucose is None:
            glucose = float(self.rng.uniform(90.0, 140.0))
        self.G = glucose / 18.0 / p.VG / p.BW  # convert mg/dL to mmol/L/kg volume
        self.Qsto = 0.0
        self.Qgut = 0.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0
        self.S1 = 0.0
        self.S2 = 0.0
        self.prev_glucose = glucose
        self.active_carbs = 0.0
        self.steps = 0
        self.state = self._build_state(glucose, 0.0, self.active_carbs)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action < 0 or action >= len(self.params.insulin_action_levels):
            raise ValueError(f"Action must be in [0, {len(self.params.insulin_action_levels)-1}]")
        p = self.params
        self.steps += 1

        # Meal: simple stochastic meal to mirror toy env
        meal_carbs = 0.0
        if self.rng.random() < 0.05:
            meal_carbs = float(self.rng.uniform(10.0, 80.0))
        self.active_carbs += meal_carbs

        # Bolus
        bolus_units = p.insulin_action_levels[action]
        self.S1 += bolus_units

        dt = p.dt

        # SC insulin absorption
        dS1 = -p.kt * self.S1
        dS2 = p.kt * self.S1 - p.kt * self.S2
        self.S1 += dS1 * dt
        self.S2 += dS2 * dt
        I = self.S2 * p.kabs / (p.VI * p.BW)  # plasma insulin (mU/L)

        # Insulin action compartments
        dx1 = -p.ka1 * self.x1 + I
        dx2 = -p.ka2 * self.x2 + I
        dx3 = -p.ka3 * self.x3 + I
        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        self.x3 += dx3 * dt

        # Gut absorption
        dQsto = -self.Qsto / p.tau_g + self.active_carbs * p.AG / p.tau_g
        dQgut = self.Qsto / p.tau_g - self.Qgut / p.tau_g
        self.Qsto += dQsto * dt
        self.Qgut += dQgut * dt
        Ra = self.Qgut / p.tau_g  # rate of appearance (g/min)
        Ra_mmol = Ra / 180.0 * 1000.0  # g -> mmol

        # Glucose dynamics (accessible compartment)
        EGP = max(p.EGP0 - self.x3, 0.0)
        F01c = min(p.F01, self.G)
        dG = (
            -F01c
            - self.x1 * self.G
            - self.x2 * self.G
            + EGP
            + Ra_mmol / (p.VG * p.BW)
        )
        self.G += dG * dt

        # Active carbs decay (for state)
        self.active_carbs = max(0.0, self.active_carbs - Ra * dt)

        glucose_mgdl = max(10.0, self.G * p.VG * p.BW * 18.0 + float(self.rng.normal(0, p.noise_std)))
        delta_glucose = glucose_mgdl - self.prev_glucose
        self.prev_glucose = glucose_mgdl

        reward = self._compute_reward(glucose_mgdl, action)
        done = (
            glucose_mgdl <= p.min_glucose
            or glucose_mgdl >= p.max_glucose
            or self.steps >= p.max_steps
        )

        self.state = self._build_state(glucose_mgdl, delta_glucose, self.active_carbs)
        info = {"glucose": glucose_mgdl, "meal": meal_carbs, "active_carbs": self.active_carbs}
        return self.state.copy(), reward, done, info

    def _compute_reward(self, glucose: float, action: int) -> float:
        p = self.params
        deviation = abs(glucose - p.target_glucose)
        reward = -deviation / 50.0
        # In-range bonus
        if 90 <= glucose <= 140:
            reward += 1.0
        # Trend term: discourage rises when high, drops when low
        # (delta glucose encoded in state; approximate with deviation change)
        # Hypo penalties
        if glucose < 70:
            reward -= 6.0
        elif glucose < 80:
            reward -= 3.0
        # Hyper penalties with ramp
        if glucose > 180:
            reward -= 2.0
        if glucose > 250:
            reward -= 3.0
        if glucose > 300:
            reward -= 4.0
        # Discourage bolus when already below target
        if glucose < p.target_glucose and action > 0:
            reward -= 1.0
        reward -= 0.01 * action  # slight action cost
        return reward

    def _build_state(self, glucose: float, delta_glucose: float, carbs: float) -> np.ndarray:
        p = self.params
        return np.array(
            [
                glucose / p.glucose_scale,
                delta_glucose / p.delta_glucose_scale,
                carbs / p.carb_scale,
            ],
            dtype=np.float32,
        )


__all__ = ["HovorkaEnv", "HovorkaParams"]
