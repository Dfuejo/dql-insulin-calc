from __future__ import annotations

import pathlib
from typing import Iterable, List, Sequence


def plot_glucose_traces(
    traces: Sequence[Iterable[float]],
    target_low: float,
    target_high: float,
    output_path: str | pathlib.Path,
    max_traces: int = 8,
    aggregate: bool = False,
) -> pathlib.Path:
    """
    Save a plot of glucose trajectories with target range shading.
    When many episodes are provided, only the first `max_traces` are drawn unless aggregate=True,
    in which case a single mean line with std shading is plotted.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt  # type: ignore  # lazy import
    import numpy as np

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    if aggregate and traces:
        # Pad traces to same length with NaN, then compute mean/std ignoring NaN
        arrays = [np.array(list(t), dtype=float) for t in traces]
        max_len = max(a.shape[0] for a in arrays)
        padded = np.full((len(arrays), max_len), np.nan, dtype=float)
        for i, a in enumerate(arrays):
            padded[i, : a.shape[0]] = a
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        plt.plot(x, mean, label="Mean glucose", color="blue")
        plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.2, label="±1 std")
        plt.title("Mean Glucose (with variance)")
    else:
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
