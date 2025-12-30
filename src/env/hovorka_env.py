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
from typing import Dict, List, Optional, Tuple

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
    max_steps: int = 288  # 24h at 5-min steps (can extend for longer horizons)
    target_glucose: float = 110.0  # mg/dL
    min_glucose: float = 40.0
    max_glucose: float = 400.0
    noise_std: float = 2.0  # mg/dL measurement noise

    # Micro-corrections to avoid large swings
    insulin_action_levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)  # U bolus options

    # Normalization for state output
    glucose_scale: float = 300.0
    delta_glucose_scale: float = 30.0
    carb_scale: float = 120.0


class HovorkaEnv:
    def __init__(
        self,
        params: HovorkaParams | None = None,
        seed: int | None = None,
        population_params: Optional[List[HovorkaParams]] = None,
    ):
        self.params = params or HovorkaParams()
        self.population_params = population_params
        self.rng = np.random.default_rng(seed)
        self.action_dim = len(self.params.insulin_action_levels)
        self.reset()

    def _prepare_meal_schedule(self) -> None:
        """Pre-sample one day's worth of meals (with jitter) to avoid per-step resampling."""
        p = self.params
        base_meals = [
            (8 * 60, 40.0),   # Breakfast
            (12 * 60, 80.0),  # Lunch
            (18 * 60, 60.0),  # Dinner
            (22 * 60, 30.0),  # Supper
        ]
        jitter = 30  # minutes early/late
        carb_noise = 20.0  # g
        steps_per_day = int(24 * 60 / p.dt)
        days = max(1, int(np.ceil(p.max_steps / steps_per_day)))
        self.meal_schedule = []
        for day in range(days):
            offset = day * steps_per_day
            for t_min, base in base_meals:
                t = (t_min + float(self.rng.uniform(-jitter, jitter))) % (24 * 60)
                carbs = max(0.0, base + float(self.rng.uniform(-carb_noise, carb_noise)))
                step_idx = int(t // p.dt) + offset
                self.meal_schedule.append({"step": step_idx, "carbs": carbs, "delivered": False})
        self.meal_schedule.sort(key=lambda m: m["step"])
        self.next_meal_idx = 0

    def reset(self, glucose: float | None = None) -> np.ndarray:
        if self.population_params:
            self.params = self.rng.choice(self.population_params)
            self.action_dim = len(self.params.insulin_action_levels)
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
        self._prepare_meal_schedule()
        self.state = self._build_state(glucose, 0.0, self.active_carbs)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action < 0 or action >= len(self.params.insulin_action_levels):
            raise ValueError(f"Action must be in [0, {len(self.params.insulin_action_levels)-1}]")
        p = self.params
        current_step = self.steps
        self.steps += 1

        # Meal schedule (pre-sampled in reset)
        meal_carbs = 0.0
        while self.next_meal_idx < len(self.meal_schedule):
            meal = self.meal_schedule[self.next_meal_idx]
            if meal["delivered"] or meal["step"] > current_step:
                break
            meal_carbs = meal["carbs"]
            meal["delivered"] = True
            self.next_meal_idx += 1
            break
        self.active_carbs += meal_carbs

        # Safety gate: suppress bolus when already low, cap small doses
        glucose_now = self.prev_glucose
        safe_action = action
        if glucose_now < 90.0:
            safe_action = 0
        elif glucose_now < 140.0:
            safe_action = min(action, 1)  # up to 0.1 U
        elif glucose_now < 200.0:
            safe_action = min(action, 2)  # up to 0.2 U
        # else allow up to 0.4 U

        bolus_units = p.insulin_action_levels[safe_action]
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

        glucose_raw = self.G * p.VG * p.BW * 18.0 + float(self.rng.normal(0, p.noise_std))
        glucose_mgdl = float(np.clip(glucose_raw, 1.0, p.max_glucose))
        delta_glucose = glucose_mgdl - self.prev_glucose
        self.prev_glucose = glucose_mgdl

        reward = self._compute_reward(glucose_mgdl, action, delta_glucose)
        done = (
            glucose_mgdl <= p.min_glucose
            or glucose_mgdl >= p.max_glucose
            or self.steps >= p.max_steps
        )

        self.state = self._build_state(glucose_mgdl, delta_glucose, self.active_carbs)
        info = {"glucose": glucose_mgdl, "meal": meal_carbs, "active_carbs": self.active_carbs}
        return self.state.copy(), reward, done, info

    def _compute_reward(self, glucose: float, action: int, delta_glucose: float) -> float:
        """
        Asymmetric reward: strongly punish hypo, mildly punish hyper,
        reward in-range, and penalize trends in the wrong direction.
        """
        p = self.params
        deviation = abs(glucose - p.target_glucose)
        reward = -deviation / 80.0  # base deviation cost softened

        # In-range bonus
        if 90 <= glucose <= 140:
            reward += 3.0
        elif 80 <= glucose <= 160:
            reward += 1.0

        # Trend penalties only when moving the wrong way
        if glucose > 180 and delta_glucose > 0:
            reward -= 1.0
        if glucose < 90 and delta_glucose < 0:
            reward -= 3.0

        # Hypo penalties (strong)
        if glucose < 70:
            reward -= 20.0
        elif glucose < 80:
            reward -= 8.0

        # Hyper penalties (softer)
        if glucose > 180:
            reward -= 0.5
        if glucose > 250:
            reward -= 1.0
        if glucose > 300:
            reward -= 2.0

        # Action cost: discourage insulin when already below target
        if glucose < p.target_glucose:
            reward -= 0.2 * action
        else:
            reward -= 0.005 * action
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
