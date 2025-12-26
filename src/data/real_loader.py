from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RealDatasetConfig:
    """Configuration for loading real CGM/meal/insulin datasets from CSV files."""

    time_col: str = "timestamp"
    glucose_col: str = "glucose_mg_dl"
    carbs_col: str = "carbs_g"  # optional; set to "" if unavailable
    insulin_col: str = "insulin_units"  # optional; set to "" if unavailable
    time_format: Optional[str] = None  # e.g., "%Y-%m-%d %H:%M:%S"
    timestep_minutes: float = 5.0  # assumed step if timestamps missing/invalid
    max_gap_minutes: float = 30.0  # start new episode when gap exceeds this
    carb_absorption_per_min: float = 0.01  # fraction cleared per minute (~50% over 70 min)


def _parse_time(raw: str, cfg: RealDatasetConfig, fallback: dt.datetime) -> dt.datetime:
    if not raw:
        return fallback
    try:
        if cfg.time_format:
            return dt.datetime.strptime(raw, cfg.time_format)
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return fallback


def load_cgm_csv(path: str | Path, cfg: Optional[RealDatasetConfig] = None) -> List[Dict]:
    """
    Load a CGM/meal/insulin CSV into episodes of dict records.
    Expects columns defined in RealDatasetConfig; missing carbs/insulin are tolerated.
    Output: list of episodes; each episode is a list of dicts with keys:
        timestamp (datetime), glucose, dG, carbs, insulin, active_carbs
    """
    cfg = cfg or RealDatasetConfig()
    path = Path(path)
    rows: List[Dict] = []

    with path.open() as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                glucose = float(row.get(cfg.glucose_col, "") or "nan")
            except ValueError:
                continue
            if math.isnan(glucose):
                continue
            carbs = float(row.get(cfg.carbs_col, "") or 0.0) if cfg.carbs_col else 0.0
            insulin = float(row.get(cfg.insulin_col, "") or 0.0) if cfg.insulin_col else 0.0
            timestamp = _parse_time(
                row.get(cfg.time_col, "") or "",
                cfg,
                fallback=dt.datetime.fromtimestamp(0),
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "glucose": glucose,
                    "carbs": carbs,
                    "insulin": insulin,
                }
            )

    # Sort by time and segment into episodes by gaps
    rows.sort(key=lambda r: r["timestamp"])
    episodes: List[List[Dict]] = []
    current: List[Dict] = []
    prev_time: Optional[dt.datetime] = None
    active_carbs = 0.0
    prev_glucose = None

    for r in rows:
        if prev_time is not None:
            delta_min = (r["timestamp"] - prev_time).total_seconds() / 60.0
            if delta_min > cfg.max_gap_minutes:
                if current:
                    episodes.append(current)
                current = []
                active_carbs = 0.0
                prev_glucose = None
            decay = math.exp(-cfg.carb_absorption_per_min * max(delta_min, 0.0))
            active_carbs *= decay
        else:
            delta_min = cfg.timestep_minutes

        active_carbs += r["carbs"]
        dG = 0.0 if prev_glucose is None else r["glucose"] - prev_glucose
        rec = {
            "timestamp": r["timestamp"],
            "glucose": r["glucose"],
            "dG": dG,
            "carbs": r["carbs"],
            "insulin": r["insulin"],
            "active_carbs": active_carbs,
            "delta_min": delta_min,
        }
        current.append(rec)
        prev_time = r["timestamp"]
        prev_glucose = r["glucose"]

    if current:
        episodes.append(current)

    return episodes


__all__ = ["RealDatasetConfig", "load_cgm_csv"]
