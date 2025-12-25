from __future__ import annotations

from typing import Dict, Iterable, List


def compute_range_metrics(
    glucose_trace: Iterable[float], target_low: float = 70.0, target_high: float = 180.0
) -> Dict[str, float]:
    """
    Compute time-in-range metrics from a sequence of glucose readings.
    Returns fractions (0.0-1.0) for:
        - tir: Time In Range   (target_low <= G <= target_high)
        - tbr: Time Below Range (G < target_low)
        - tor: Time Over Range  (G > target_high)
    """
    values: List[float] = list(glucose_trace)
    if not values:
        return {"tir": 0.0, "tbr": 0.0, "tor": 0.0}

    total = float(len(values))
    below = sum(g < target_low for g in values)
    above = sum(g > target_high for g in values)
    in_range = total - below - above

    return {
        "tir": in_range / total,
        "tbr": below / total,
        "tor": above / total,
    }


__all__ = ["compute_range_metrics"]
