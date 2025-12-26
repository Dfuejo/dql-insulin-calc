from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import xml.etree.ElementTree as ET


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


def _parse_ts(ts: str) -> dt.datetime:
    # Ohio dataset format: dd-mm-YYYY HH:MM:SS
    return dt.datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")


def load_ohio_xml(
    path: str | Path, max_gap_minutes: float = 30.0, carb_absorption_per_min: float = 0.01
) -> List[List[Dict]]:
    """
    Load OhioT1DM-style XML (glucose_level, meal, bolus sections) into episodes.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    glucose_events = []
    for ev in root.findall("./glucose_level/event"):
        ts = ev.get("ts")
        val = ev.get("value")
        if ts is None or val is None:
            continue
        glucose_events.append({"timestamp": _parse_ts(ts), "glucose": float(val)})
    glucose_events.sort(key=lambda r: r["timestamp"])

    meal_events = []
    for ev in root.findall("./meal/event"):
        ts = ev.get("ts")
        carbs = ev.get("carbs")
        if ts is None or carbs is None:
            continue
        meal_events.append({"timestamp": _parse_ts(ts), "carbs": float(carbs)})
    meal_events.sort(key=lambda r: r["timestamp"])

    bolus_events = []
    for ev in root.findall("./bolus/event"):
        ts = ev.get("ts_begin") or ev.get("ts")
        dose = ev.get("dose")
        if ts is None or dose is None:
            continue
        bolus_events.append({"timestamp": _parse_ts(ts), "insulin": float(dose)})
    bolus_events.sort(key=lambda r: r["timestamp"])

    episodes: List[List[Dict]] = []
    current: List[Dict] = []
    prev_time: Optional[dt.datetime] = None
    prev_glucose: Optional[float] = None
    active_carbs = 0.0

    meal_idx = 0
    bolus_idx = 0

    for g in glucose_events:
        ts = g["timestamp"]
        if prev_time is not None:
            delta_min = (ts - prev_time).total_seconds() / 60.0
            if delta_min > max_gap_minutes:
                if current:
                    episodes.append(current)
                current = []
                active_carbs = 0.0
                prev_glucose = None
            decay = math.exp(-carb_absorption_per_min * max(delta_min, 0.0))
            active_carbs *= decay
        else:
            delta_min = 5.0

        while meal_idx < len(meal_events) and meal_events[meal_idx]["timestamp"] <= ts:
            active_carbs += meal_events[meal_idx]["carbs"]
            meal_idx += 1

        insulin = 0.0
        while bolus_idx < len(bolus_events) and bolus_events[bolus_idx]["timestamp"] <= ts:
            insulin += bolus_events[bolus_idx]["insulin"]
            bolus_idx += 1

        glucose = g["glucose"]
        dG = 0.0 if prev_glucose is None else glucose - prev_glucose

        rec = {
            "timestamp": ts,
            "glucose": glucose,
            "dG": dG,
            "carbs": 0.0,
            "insulin": insulin,
            "active_carbs": active_carbs,
            "delta_min": delta_min,
        }
        current.append(rec)
        prev_time = ts
        prev_glucose = glucose

    if current:
        episodes.append(current)

    return episodes


__all__ = ["RealDatasetConfig", "load_cgm_csv", "load_ohio_xml"]
