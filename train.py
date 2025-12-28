"""
Lightweight training harness for the DQN.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import replace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from agents import DQNConfig, evaluate_policy, plot_glucose_traces, train_dqn, save_policy  # noqa: E402
from env import EnvParams, InsulinEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN for insulin control.")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes after training.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for environment.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a short smoke-test training (fewer episodes, smaller warmup).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="plots/glucose_eval.png",
        help="Path to save evaluation glucose trace plot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device preference. 'auto' tries mps->cuda->cpu.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Optional path to save policy network state_dict after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Environment configuration
    env_params = EnvParams()
    env = InsulinEnv(env_params, seed=args.seed)

    # DQN configuration (tweak for speed vs performance)
    config = DQNConfig(
        max_episodes=args.episodes,
        batch_size=64,
        hidden_sizes=(256, 256),
        buffer_size=75_000,
    )

    if args.fast:
        config = replace(
            config,
            max_episodes=min(30, config.max_episodes),
            min_buffer_size=500,
            buffer_size=20_000,
        )

    # Device selection
    try:
        import torch

        if args.device == "mps":
            config = replace(config, device="mps")
        elif args.device == "cuda":
            config = replace(config, device="cuda")
        elif args.device == "cpu":
            config = replace(config, device="cpu")
        else:  # auto
            if torch.backends.mps.is_available():
                config = replace(config, device="mps")
            elif torch.cuda.is_available():
                config = replace(config, device="cuda")
    except ImportError:
        pass

    agent, history = train_dqn(env, config)
    mean_last10 = sum(history["episode_rewards"][-10:]) / min(10, len(history["episode_rewards"]))
    print(
        f"Finished training. Episodes: {len(history['episode_rewards'])}, "
        f"mean reward (last 10): {mean_last10:.3f}"
    )

    eval_results = evaluate_policy(
        env,
        agent,
        episodes=args.eval_episodes,
        target_low=config.target_low,
        target_high=config.target_high,
    )
    mm = eval_results["mean_metrics"]
    mean_eval_reward = sum(eval_results["episode_rewards"]) / max(1, len(eval_results["episode_rewards"]))
    print(
        f"Evaluation over {args.eval_episodes} episodes "
        f"(epsilon=0): reward={mean_eval_reward:.3f}, "
        f"TIR={mm['tir']:.2f}, TBR={mm['tbr']:.2f}, TOR={mm['tor']:.2f}"
    )

    # Plot glucose traces if matplotlib is available
    try:
        output_path = plot_glucose_traces(
            eval_results["glucose_traces"], config.target_low, config.target_high, args.plot_path, aggregate=True
        )
        print(f"Saved evaluation glucose plot to {output_path}")
    except ImportError:
        print("matplotlib not installed; skipping glucose plot.")

    if args.save_checkpoint:
        save_policy(agent, args.save_checkpoint)
        print(f"Saved policy checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
