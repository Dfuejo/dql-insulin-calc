"""
Simulate 120 episodes in Hovorka with:
 - No agent (no bolus baseline)
 - Constant policies (Baseline I and II: fixed small bolus each step)
Reports TIR/TBR/TOR and saves glucose plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from env import HovorkaEnv, HovorkaParams  # noqa: E402
from agents.metrics import compute_range_metrics  # noqa: E402
from agents.plotting import plot_glucose_traces  # noqa: E402


def run_policy(env: HovorkaEnv, episodes: int, action_fn, name: str):
    traces = []
    metrics = []
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        trace = []
        total_reward = 0.0
        for _ in range(env.params.max_steps):
            action = action_fn(state)
            next_state, reward, done, info = env.step(action)
            trace.append(float(info.get("glucose", 0.0)))
            total_reward += reward
            state = next_state
            if done:
                break
        m = compute_range_metrics(trace, target_low=env.params.target_glucose - 40, target_high=env.params.target_glucose + 70)
        traces.append(trace)
        metrics.append(m)
        rewards.append(total_reward)
    mean_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    return {"name": name, "traces": traces, "metrics": mean_metrics, "rewards": rewards}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate baselines without RL agent.")
    parser.add_argument("--episodes", type=int, default=120, help="Number of episodes to simulate per policy.")
    parser.add_argument("--output", type=str, default="plots/baseline_policies.png", help="Path to save overlay plot.")
    args = parser.parse_args()

    env = HovorkaEnv(HovorkaParams(), seed=0)

    # Baseline policies
    no_action = lambda state: 0
    constant_small = lambda state: 1  # 0.1 U with current action set
    constant_medium = lambda state: 2  # 0.2 U

    results = [
        run_policy(env, args.episodes, no_action, "No action"),
        run_policy(env, args.episodes, constant_small, "Const 0.1U"),
        run_policy(env, args.episodes, constant_medium, "Const 0.2U"),
    ]

    for r in results:
        m = r["metrics"]
        print(f"{r['name']}: TIR={m['tir']:.2f}, TBR={m['tbr']:.2f}, TOR={m['tor']:.2f}, mean reward={np.mean(r['rewards']):.2f}")

    try:
        plot_glucose_traces(
            results[0]["traces"],
            target_low=70,
            target_high=180,
            output_path=args.output,
            aggregate=True,
            overlay=[res["traces"] for res in results[1:]],
            overlay_labels=[res["name"] for res in results[1:]],
        )
        print(f"Saved overlay plot to {args.output}")
    except ImportError:
        print("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
