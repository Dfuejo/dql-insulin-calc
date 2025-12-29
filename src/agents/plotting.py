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
    overlay: Sequence[Sequence[Iterable[float]]] | None = None,
    overlay_labels: Sequence[str] | None = None,
) -> pathlib.Path:
    """
    Save a plot of glucose trajectories with target range shading.
    When many episodes are provided, only the first `max_traces` are drawn unless aggregate=True,
    in which case a single mean line with std shading is plotted.
    If overlay is provided, it should be a list of trace sets to plot alongside the primary traces.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt  # type: ignore  # lazy import
    import numpy as np

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.style.use("seaborn-v0_8-muted")
    plt.grid(True, linestyle="--", alpha=0.3)

    def plot_set(data: Sequence[Iterable[float]], label_prefix: str, color: str | None = None):
        if aggregate and data:
            arrays = [np.array(list(t), dtype=float) for t in data]
            max_len = max(a.shape[0] for a in arrays)
            padded = np.full((len(arrays), max_len), np.nan, dtype=float)
            for i, a in enumerate(arrays):
                padded[i, : a.shape[0]] = a
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            x = np.arange(max_len)
            plt.plot(x, mean, label=f"{label_prefix} mean", color=color)
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, label=f"{label_prefix} ±1 std")
        else:
            for idx, trace in enumerate(data[:max_traces]):
                plt.plot(list(trace), alpha=0.6, label=f"{label_prefix} ep {idx+1}", color=color)

    if traces:
        # Pad traces to same length with NaN, then compute mean/std ignoring NaN
        plot_set(traces, "Policy", color="C0")
    else:
        plt.title("No traces provided")

    if overlay:
        labels = overlay_labels or [f"Overlay {i+1}" for i in range(len(overlay))]
        for data, lbl, color in zip(overlay, labels, plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]):
            plot_set(data, lbl, color=color)

    plt.axhspan(target_low, target_high, color="green", alpha=0.1, label="Target range")
    plt.xlabel("Step")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


__all__ = ["plot_glucose_traces"]
