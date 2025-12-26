from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


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


def excursion_stats(
    glucose_trace: Iterable[float], target_low: float = 70.0, target_high: float = 180.0
) -> Dict[str, float]:
    """
    Compute min/max, duration of longest over/under excursions (in steps),
    and total excursion counts.
    """
    values: List[float] = list(glucose_trace)
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "longest_under": 0,
            "longest_over": 0,
            "count_under": 0,
            "count_over": 0,
        }

    def longest_and_count(pred) -> Tuple[int, int]:
        longest = 0
        current = 0
        count = 0
        for v in values:
            if pred(v):
                current += 1
            else:
                if current > 0:
                    count += 1
                    longest = max(longest, current)
                current = 0
        if current > 0:
            count += 1
            longest = max(longest, current)
        return longest, count

    longest_under, count_under = longest_and_count(lambda g: g < target_low)
    longest_over, count_over = longest_and_count(lambda g: g > target_high)

    return {
        "min": min(values),
        "max": max(values),
        "longest_under": float(longest_under),
        "longest_over": float(longest_over),
        "count_under": float(count_under),
        "count_over": float(count_over),
    }


__all__ = ["compute_range_metrics", "excursion_stats"]
