# RL Insulin Calculator

Reinforcement Learning project for insulin decision-making in Type 1 Diabetes
using synthetic environments (toy and Hovorka).

## Problem setup

- **State:** `(G, dG, CH)` where `G` is current glucose (mg/dL), `dG` is change
  since the previous step, and `CH` is active carbohydrates (grams). Optionally
  stack recent observations via `state_history` (default 1; see `DQNConfig`).
- **Actions:** Toy env: `[0, 1]` (no bolus / bolus). Hovorka env: 4 bolus levels
  (0, 0.5U, 1U, 2U) to limit catastrophic hypos while allowing corrections.
- **Environments:**
  - `InsulinEnv` — fast toy dynamics.
  - `HovorkaEnv` — discrete-time Hovorka model (5‑min steps) with multi-action
    dosing, pre-sampled meals (4 per day with timing and carb noise), and an
    asymmetric reward (fort hypo, suau hyper, bonificació en rang, penalitzacions
    de tendència només quan el canvi és desfavorable). Select via `--env toy|hovorka`.
- **Network/agent:** Double DQN with dueling heads, n-step returns, optional
  prioritized replay (PER), warm-up heurístic inicial per omplir el buffer, i
  decay exponencial d’epsilon (eps_0=1.0, eps_f=0.05, eta=50k passos).

## Quick start

1) Install dependencies (CPU Torch is fine for initial experimentation):
   ```bash
   pip install torch numpy matplotlib
   ```
   - On Apple Silicon (M1/M2), the default PyPI wheel enables Metal (MPS) acceleration.
2) Train the agent (uses sensible defaults):
   ```bash
   python train.py
   ```
   - Add `--fast` for a quick smoke test (fewer episodes, smaller buffer).
   - Use `--episodes`/`--eval-episodes` to scale duration and evaluation runs.
   - Use `--plot-path` to save glucose plots after evaluation (requires matplotlib).
   - Use `--save-checkpoint policy.pt` to persist the trained policy.
   - Use `--device cpu|mps|cuda|auto` to pick compute device (default: auto).
   - Use `--env hovorka` to train on the Hovorka model; `--hovorka-patient`
     can be `adult|adolescent|child|random` to sample presets.
3) Tune hyperparameters in `train.py` or via the dataclasses in
   `src/env/insulin_env.py` (`EnvParams`) and `src/agents/dqn_agent.py`
   (`DQNConfig`) to scale up/down for performance and fidelity.

## Metrics

- Reports Time In Range (TIR), Time Below Range (TBR), and Time Over Range (TOR)
  using thresholds from `DQNConfig.target_low`/`target_high` (defaults 70/180).
- Training logs rolling TIR/TBR/TOR and reward variance per bloc.
- After training, an evaluation run with epsilon=0 prints mean TIR/TBR/TOR and
  reward across eval episodes, and can save glucose traces. Plots can aggregate
  traces into a mean line with ±1 std shading.
- `baseline_eval.py` compares the trained policy against a simple heuristic
  baseline and can overlay plots.

## Real-world data (offline eval)

- Expected CSV columns (rename in `RealDatasetConfig` if needed):
  - `timestamp` (ISO or with a format string)
  - `glucose_mg_dl` (required)
  - `carbs_g` (optional)
  - `insulin_units` (optional)
- Run offline evaluation on real logs:
  ```bash
  python evaluate_real.py --csv path/to/data.csv --plot-path plots/real_eval.png
  ```
- To compare your trained policy in simulation (counterfactual rollout), save a checkpoint
  during training (`--save-checkpoint policy.pt`), then:
  ```bash
  python evaluate_real.py --csv path/to/data.csv --checkpoint policy.pt --rollout-episodes 5 --sim-plot-path plots/real_sim_eval.png
  ```
- OhioT1DM XML is supported with `--format ohio-xml` (auto-detected on `.xml`):
  ```bash
  python evaluate_real.py --csv datasets/archiveOhio/559-ws-training.xml --format ohio-xml --plot-path plots/ohio_eval.png
  ```

## Code layout

- `src/env/insulin_env.py` — lightweight insulin/glucose environment (toy).
- `src/env/hovorka_env.py` — Hovorka-based environment with multi-level insulin actions.
- `src/agents/dqn_agent.py` — DQN agent, Q-network, and training loop.
- `src/agents/replay_buffer.py` — experience buffer for off-policy learning.
- `train.py` — runnable training harness with tweakable defaults.
