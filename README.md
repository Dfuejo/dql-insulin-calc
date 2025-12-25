# RL Insulin Calculator

Reinforcement Learning project for insulin decision-making in Type 1 Diabetes
using a synthetic environment.

## Problem setup

- **State:** `(G, dG, CH)` where `G` is current glucose (mg/dL), `dG` is change
  since the previous step, and `CH` is active carbohydrates (grams) being
  absorbed.
- **Actions:** `[0, 1]` → `0` does nothing, `1` delivers a fixed rapid-acting
  insulin bolus.
- **Network:** a small MLP that outputs Q-values for each action.
- **Environment:** stochastic meal generation, simple glucose/carbohydrate and
  insulin-on-board dynamics; fast enough for a laptop but configurable for
  more realistic simulations.

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
   - Use `--device cpu|mps|cuda|auto` to pick compute device (default: auto).
3) Tune hyperparameters in `train.py` or via the dataclasses in
   `src/env/insulin_env.py` (`EnvParams`) and `src/agents/dqn_agent.py`
   (`DQNConfig`) to scale up/down for performance and fidelity.

## Metrics

- Reports Time In Range (TIR), Time Below Range (TBR), and Time Over Range (TOR)
  using thresholds from `DQNConfig.target_low`/`target_high` (defaults 70/180).
- Training logs rolling TIR/TBR/TOR; after training, an evaluation run with
  epsilon=0 prints mean TIR/TBR/TOR and reward across eval episodes, and can
  optionally save glucose traces to a plot.

## Code layout

- `src/env/insulin_env.py` — lightweight insulin/glucose environment.
- `src/agents/dqn_agent.py` — DQN agent, Q-network, and training loop.
- `src/agents/replay_buffer.py` — experience buffer for off-policy learning.
- `train.py` — runnable training harness with tweakable defaults.
