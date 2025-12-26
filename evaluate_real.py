"""
Offline evaluation script for real CGM/meal/insulin logs.

Examples:
    python evaluate_real.py --csv path/to/data.csv --episodes 20 --plot-path plots/real_eval.png
    python evaluate_real.py --csv path/to/data.csv --checkpoint policy.pt --rollout-episodes 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import replace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from agents import (  # noqa: E402
    DQNAgent,
    DQNConfig,
    compute_range_metrics,
    evaluate_policy,
    load_policy,
    plot_glucose_traces,
    excursion_stats,
)  # noqa: E402
from data import RealDatasetConfig, load_cgm_csv, load_ohio_xml  # noqa: E402
from env import EnvParams, InsulinEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation on real CGM logs.")
    parser.add_argument("--csv", required=True, help="Path to CGM/meal/insulin CSV or Ohio XML.")
    parser.add_argument("--episodes", type=int, default=None, help="Limit number of episodes to use.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (.pt). If absent, no counterfactual rollout is done.",
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=0,
        help="Number of episodes to roll out in the simulator with the loaded policy.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="plots/real_eval.png",
        help="Where to save glucose traces plot for real data.",
    )
    parser.add_argument(
        "--sim-plot-path",
        type=str,
        default="plots/real_sim_eval.png",
        help="Where to save simulated rollout traces (if checkpoint provided).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device for policy eval (if checkpoint provided).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "csv", "ohio-xml"],
        help="File format; auto infers from extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RealDatasetConfig()
    path = pathlib.Path(args.csv)
    fmt = args.format
    if fmt == "auto":
        if path.suffix.lower() == ".xml":
            fmt = "ohio-xml"
        else:
            fmt = "csv"

    if fmt == "ohio-xml":
        episodes = load_ohio_xml(path)
    else:
        episodes = load_cgm_csv(path, cfg)
    if args.episodes:
        episodes = episodes[: args.episodes]

    # Metrics on real logs
    real_metrics = []
    real_excur = []
    for ep in episodes:
        glucose_trace = [rec["glucose"] for rec in ep]
        real_metrics.append(compute_range_metrics(glucose_trace))
        real_excur.append(excursion_stats(glucose_trace))
    mean_real = {
        "tir": sum(m["tir"] for m in real_metrics) / max(1, len(real_metrics)),
        "tbr": sum(m["tbr"] for m in real_metrics) / max(1, len(real_metrics)),
        "tor": sum(m["tor"] for m in real_metrics) / max(1, len(real_metrics)),
    }
    mean_real_excur = {
        k: sum(e[k] for e in real_excur) / max(1, len(real_excur)) for k in real_excur[0]
    } if real_excur else {}
    print(
        f"Real data metrics over {len(episodes)} episodes: "
        f"TIR={mean_real['tir']:.2f}, TBR={mean_real['tbr']:.2f}, TOR={mean_real['tor']:.2f}"
    )
    if mean_real_excur:
        print(
            "Real excursion stats "
            f"(min={mean_real_excur['min']:.1f}, max={mean_real_excur['max']:.1f}, "
            f"longest_under={mean_real_excur['longest_under']:.0f} steps, "
            f"longest_over={mean_real_excur['longest_over']:.0f} steps)"
        )

    # Plot real traces
    try:
        plot_glucose_traces(
            [[rec["glucose"] for rec in ep] for ep in episodes],
            target_low=70.0,
            target_high=180.0,
            output_path=args.plot_path,
            max_traces=8,
        )
        print(f"Saved real glucose plot to {args.plot_path}")
    except ImportError:
        print("matplotlib not installed; skipping real glucose plot.")

    # Optional counterfactual rollout in simulator using checkpoint
    if args.checkpoint and args.rollout_episodes > 0:
        # Device selection
        device = None
        try:
            import torch

            if args.device == "mps":
                device = "mps"
            elif args.device == "cuda":
                device = "cuda"
            elif args.device == "cpu":
                device = "cpu"
            elif args.device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
        except ImportError:
            pass

        env_params = EnvParams()
        env = InsulinEnv(env_params)
        dqn_cfg = DQNConfig(device=device)
        agent = DQNAgent(state_dim=3, action_dim=2, config=dqn_cfg)
        load_policy(agent, args.checkpoint)

        rollout_eps = min(args.rollout_episodes, len(episodes))
        sim_metrics = []
        sim_excur = []
        sim_traces = []
        for ep in episodes[:rollout_eps]:
            # seed the env to starting glucose
            start_glucose = ep[0]["glucose"]
            state = env.reset(glucose=start_glucose)
            trace = [start_glucose]
            total_reward = 0.0
            for _ in range(len(ep) - 1):
                action = agent.act(state, epsilon=0.0)
                next_state, reward, done, info = env.step(action)
                state = next_state
                trace.append(float(info.get("glucose", 0.0)))
                total_reward += reward
                if done:
                    break
            sim_traces.append(trace)
            sim_metrics.append(compute_range_metrics(trace))
            sim_excur.append(excursion_stats(trace))

        if sim_metrics:
            mean_sim = {
                "tir": sum(m["tir"] for m in sim_metrics) / len(sim_metrics),
                "tbr": sum(m["tbr"] for m in sim_metrics) / len(sim_metrics),
                "tor": sum(m["tor"] for m in sim_metrics) / len(sim_metrics),
            }
            mean_sim_excur = {
                k: sum(e[k] for e in sim_excur) / len(sim_excur) for k in sim_excur[0]
            } if sim_excur else {}
            print(
                f"Simulated rollouts ({len(sim_metrics)} episodes): "
                f"TIR={mean_sim['tir']:.2f}, TBR={mean_sim['tbr']:.2f}, TOR={mean_sim['tor']:.2f}"
            )
            if mean_sim_excur:
                print(
                    "Simulated excursion stats "
                    f"(min={mean_sim_excur['min']:.1f}, max={mean_sim_excur['max']:.1f}, "
                    f"longest_under={mean_sim_excur['longest_under']:.0f} steps, "
                    f"longest_over={mean_sim_excur['longest_over']:.0f} steps)"
                )
            try:
                plot_glucose_traces(
                    sim_traces,
                    target_low=70.0,
                    target_high=180.0,
                    output_path=args.sim_plot_path,
                    max_traces=8,
                )
                print(f"Saved simulated glucose plot to {args.sim_plot_path}")
            except ImportError:
                print("matplotlib not installed; skipping simulated plot.")
        else:
            print("No simulated metrics computed.")


if __name__ == "__main__":
    main()
