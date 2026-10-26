"""Synthetic run-to-failure telemetry for a fleet of rotating machines.

The generator follows the spirit of public turbofan degradation benchmarks without
using any of their data. Each unit starts with a small, random amount of wear,
degrades along an accelerating curve and fails when its damage reaches 1.0. Sensors
respond to damage with their own gain and direction, on top of measurement noise.
Some sensors carry no signal at all, as in a real plant.

Everything is driven by a single seed, so the same call returns the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNIT_COL = "unit_id"
CYCLE_COL = "cycle"
TARGET_COL = "rul"
OP_SETTINGS: tuple[str, ...] = ("op_setting_1", "op_setting_2", "op_setting_3")


@dataclass(frozen=True)
class SensorSpec:
    """How one sensor responds to damage."""

    name: str
    baseline: float
    gain: float
    noise: float


# Gains are signed: a negative gain means the reading falls as the machine wears.
# The last three sensors carry no degradation signal, only noise.
SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec("sensor_01", 642.0, 6.5, 0.50),  # outlet temperature
    SensorSpec("sensor_02", 1589.0, 38.0, 6.00),  # exhaust temperature
    SensorSpec("sensor_03", 1400.0, 32.0, 9.00),  # bearing temperature
    SensorSpec("sensor_04", 554.0, -7.0, 0.90),  # discharge pressure
    SensorSpec("sensor_05", 2388.0, 0.45, 0.07),  # shaft speed
    SensorSpec("sensor_06", 47.5, 1.9, 0.27),  # static pressure ratio
    SensorSpec("sensor_07", 521.0, -5.5, 0.74),  # fuel flow ratio
    SensorSpec("sensor_08", 8.44, 0.25, 0.04),  # bypass ratio
    SensorSpec("sensor_09", 393.0, 7.0, 1.50),  # bleed enthalpy
    SensorSpec("sensor_10", 38.8, -1.3, 0.18),  # coolant flow
    SensorSpec("sensor_11", 100.0, 0.0, 0.30),  # nominal vibration, no signal
    SensorSpec("sensor_12", 21.6, 0.0, 0.15),  # ambient pressure, no signal
    SensorSpec("sensor_13", 0.03, 0.0, 0.002),  # demand signal, no signal
)
SENSOR_COLS: tuple[str, ...] = tuple(s.name for s in SENSORS)
INFORMATIVE_SENSORS: tuple[str, ...] = tuple(s.name for s in SENSORS if s.gain != 0.0)


@dataclass(frozen=True)
class FleetConfig:
    """Shape of a synthetic fleet."""

    n_units: int = 100
    min_life: int = 130
    max_life: int = 360
    seed: int = 42


def _unit_frame(unit_id: int, life: int, rng: np.random.Generator) -> pd.DataFrame:
    cycles = np.arange(1, life + 1)
    initial_wear = rng.uniform(0.0, 0.12)
    shape = rng.uniform(1.6, 3.0)
    damage = initial_wear + (1.0 - initial_wear) * (cycles / life) ** shape

    columns: dict[str, np.ndarray] = {
        UNIT_COL: np.full(life, unit_id, dtype=np.int64),
        CYCLE_COL: cycles.astype(np.int64),
        "op_setting_1": rng.normal(0.0, 0.002, life),
        "op_setting_2": rng.normal(0.0, 0.0003, life),
        "op_setting_3": np.full(life, 100.0),
    }
    unit_sensitivity = rng.normal(1.0, 0.08)
    for spec in SENSORS:
        offset = rng.normal(0.0, spec.noise * 0.5)
        signal = spec.gain * unit_sensitivity * damage
        columns[spec.name] = spec.baseline + offset + signal + rng.normal(0.0, spec.noise, life)
    columns[TARGET_COL] = (life - cycles).astype(np.int64)
    return pd.DataFrame(columns)


def generate_fleet(config: FleetConfig | None = None) -> pd.DataFrame:
    """Return run-to-failure telemetry, one row per unit and cycle, with the true RUL."""
    cfg = config or FleetConfig()
    rng = np.random.default_rng(cfg.seed)
    lives = rng.integers(cfg.min_life, cfg.max_life + 1, size=cfg.n_units)
    frames = [_unit_frame(i + 1, int(life), rng) for i, life in enumerate(lives)]
    return pd.concat(frames, ignore_index=True)


def truncate_fleet(fleet: pd.DataFrame, seed: int, min_fraction: float = 0.3) -> pd.DataFrame:
    """Cut every unit at a random cycle, as if observed in service before failure.

    The returned frame keeps the true RUL at each row, so the last row of each unit
    is the ground truth an evaluation compares against.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, unit in fleet.groupby(UNIT_COL, sort=True):
        life = len(unit)
        cut = int(rng.integers(max(int(life * min_fraction), 1), life))
        parts.append(unit.iloc[:cut])
    return pd.concat(parts, ignore_index=True)


def apply_sensor_shift(frame: pd.DataFrame, sensor: str, shift: float) -> pd.DataFrame:
    """Return a copy with a constant offset on one sensor, to simulate a calibration fault."""
    shifted = frame.copy()
    shifted[sensor] = shifted[sensor] + shift
    return shifted
