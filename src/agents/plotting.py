from __future__ import annotations

import pathlib
from typing import Iterable, List, Sequence


def plot_glucose_traces(
    traces: Sequence[Iterable[float]],
    target_low: float,
    target_high: float,
    output_path: str | pathlib.Path,
    max_traces: int = 8,
) -> pathlib.Path:
    """
    Save a plot of glucose trajectories with target range shading.
    When many episodes are provided, only the first `max_traces` are drawn.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt  # type: ignore  # lazy import

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for idx, trace in enumerate(traces[:max_traces]):
        plt.plot(list(trace), alpha=0.6, label=f"Episode {idx+1}")

    if len(traces) > max_traces:
        plt.title(f"Glucose Traces (showing {max_traces} of {len(traces)} episodes)")
    else:
        plt.title("Glucose Traces")

    plt.axhspan(target_low, target_high, color="green", alpha=0.1, label="Target range")
    plt.xlabel("Step")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


__all__ = ["plot_glucose_traces"]
