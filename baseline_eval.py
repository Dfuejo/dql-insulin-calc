"""
Baseline comparison for Hovorka: rule-based vs trained policy.

Usage:
    python baseline_eval.py --episodes 50 --checkpoint policy.pt

The baseline uses fixed basal and a simple meal bolus rule based on estimated carbs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from agents import (
    DQNAgent,
    DQNConfig,
    compute_range_metrics,
    load_policy,
    plot_glucose_traces,
)  # noqa: E402
from env import HovorkaEnv, HovorkaParams  # noqa: E402


def run_policy(env, agent: DQNAgent, episodes: int, epsilon: float) -> dict:
    metrics = []
    traces = []
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        trace = []
        total_reward = 0.0
        for _ in range(env.params.max_steps):
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            trace.append(float(info.get("glucose", 0.0)))
            total_reward += reward
            state = next_state
            if done:
                break
        m = compute_range_metrics(trace)
        metrics.append(m)
        traces.append(trace)
        rewards.append(total_reward)
    mean_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    mean_reward = float(np.mean(rewards))
    return {"metrics": mean_metrics, "rewards": rewards, "mean_reward": mean_reward, "traces": traces}


def rule_based_agent(env: HovorkaEnv):
    """Simple bolus rule: choose action proportional to current glucose above target."""
    class RuleAgent:
        def __init__(self, action_levels):
            self.action_levels = action_levels
        def act(self, state, epsilon=0.0):
            glucose = state[0] * env.params.glucose_scale
            if glucose < 140:
                return 0
            # pick a higher action for higher glucose
            if glucose > 300:
                return len(self.action_levels) - 1
            if glucose > 220:
                return max(0, len(self.action_levels) - 2)
            if glucose > 180:
                return max(0, len(self.action_levels) - 3)
            return 1  # mild correction
    return RuleAgent(env.params.insulin_action_levels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs trained policy on Hovorka.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for evaluation.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy checkpoint.")
    parser.add_argument("--plot-path", type=str, default=None, help="Path to save overlay plot.")
    args = parser.parse_args()

    env = HovorkaEnv(HovorkaParams(), seed=0)

    # Baseline
    baseline_agent = rule_based_agent(env)
    baseline = run_policy(env, baseline_agent, args.episodes, epsilon=0.0)

    # Trained policy (if provided)
    trained = None
    if args.checkpoint:
        cfg = DQNConfig()
        agent = DQNAgent(state_dim=3, action_dim=env.action_dim, config=cfg)
        load_policy(agent, args.checkpoint)
        trained = run_policy(env, agent, args.episodes, epsilon=0.0)

    print(
        f"Baseline ({args.episodes} eps): reward={baseline['mean_reward']:.2f}, "
        f"TIR={baseline['metrics']['tir']:.2f}, TBR={baseline['metrics']['tbr']:.2f}, TOR={baseline['metrics']['tor']:.2f}"
    )
    if trained:
        print(
            f"Trained  ({args.episodes} eps): reward={trained['mean_reward']:.2f}, "
            f"TIR={trained['metrics']['tir']:.2f}, TBR={trained['metrics']['tbr']:.2f}, TOR={trained['metrics']['tor']:.2f}"
        )

    if args.plot_path and trained:
        try:
            plot_glucose_traces(
                trained["traces"],
                target_low=70.0,
                target_high=180.0,
                output_path=args.plot_path,
                aggregate=True,
                overlay=[baseline["traces"]],
                overlay_labels=["Baseline"],
            )
            print(f"Saved overlay plot to {args.plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
