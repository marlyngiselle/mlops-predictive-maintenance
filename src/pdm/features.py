"""Feature engineering for remaining useful life.

All features are computed per unit, in cycle order, and only look backwards, so the
same function serves training, batch scoring and the online endpoint without leakage.

Feature families:

* rolling mean and standard deviation over a short window
* rolling slope (least squares trend) over a long window
* deviation from the unit's own early life baseline
* a health indicator: the mean of signed, noise scaled baseline deviations of informative sensors
* lags of the health indicator, to expose the recent rate of change
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pdm.data import CYCLE_COL, INFORMATIVE_SENSORS, SENSOR_COLS, SENSORS, TARGET_COL, UNIT_COL

RUL_CAP = 125
# Direction of wear per sensor. In a plant this comes from the equipment engineers,
# not from the data: it says whether a rising reading is good news or bad news.
WEAR_DIRECTION = {s.name: float(np.sign(s.gain)) for s in SENSORS if s.gain != 0.0}


@dataclass(frozen=True)
class FeatureConfig:
    """Window sizes and sensor lists. Logged with every run for lineage."""

    sensors: tuple[str, ...] = SENSOR_COLS
    short_window: int = 10
    long_window: int = 30
    baseline_cycles: int = 15
    hi_lags: tuple[int, ...] = (5, 15)
    informative: tuple[str, ...] = field(default=INFORMATIVE_SENSORS)

    def as_params(self) -> dict[str, str]:
        return {
            "fe_short_window": str(self.short_window),
            "fe_long_window": str(self.long_window),
            "fe_baseline_cycles": str(self.baseline_cycles),
            "fe_hi_lags": ",".join(str(x) for x in self.hi_lags),
            "fe_n_sensors": str(len(self.sensors)),
        }


def _rolling(frame: pd.DataFrame, cols: Sequence[str], window: int, how: str) -> pd.DataFrame:
    grouped = frame.groupby(UNIT_COL, sort=False)[list(cols)]
    rolled = grouped.rolling(window, min_periods=1)
    out = rolled.mean() if how == "mean" else rolled.std(ddof=0)
    return out.reset_index(level=0, drop=True).sort_index()


def _rolling_slope(frame: pd.DataFrame, cols: Sequence[str], window: int) -> pd.DataFrame:
    """Least squares slope per cycle, from rolling moments (no Python loop per row)."""
    t = frame[CYCLE_COL].astype(float)
    work = pd.DataFrame({UNIT_COL: frame[UNIT_COL], "_t": t, "_tt": t * t}, index=frame.index)
    for c in cols:
        work[c] = frame[c]
        work[f"_tx_{c}"] = t * frame[c]
    means = _rolling(work, [c for c in work.columns if c != UNIT_COL], window, "mean")
    var_t = means["_tt"] - means["_t"] ** 2
    slopes = {}
    for c in cols:
        cov = means[f"_tx_{c}"] - means["_t"] * means[c]
        slopes[c] = np.where(var_t > 1e-9, cov / var_t.where(var_t > 1e-9, 1.0), 0.0)
    return pd.DataFrame(slopes, index=frame.index)


def build_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return identifiers, the cycle, engineered features and, when present, the capped target."""
    cfg = config or FeatureConfig()
    df = frame.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)
    sensors = list(cfg.sensors)

    out = pd.DataFrame({UNIT_COL: df[UNIT_COL], CYCLE_COL: df[CYCLE_COL]})
    means = _rolling(df, sensors, cfg.short_window, "mean")
    stds = _rolling(df, sensors, cfg.short_window, "std")
    slopes = _rolling_slope(df, sensors, cfg.long_window)

    early = df[df[CYCLE_COL] <= df.groupby(UNIT_COL)[CYCLE_COL].transform("min") + cfg.baseline_cycles - 1]
    baseline = early.groupby(UNIT_COL)[sensors].mean()
    base_aligned = baseline.reindex(df[UNIT_COL]).set_index(df.index)
    base_spread = early.groupby(UNIT_COL)[sensors].std(ddof=0).clip(lower=1e-6)
    spread_aligned = base_spread.reindex(df[UNIT_COL]).set_index(df.index)

    for c in sensors:
        out[f"{c}_mean_{cfg.short_window}"] = means[c]
        out[f"{c}_std_{cfg.short_window}"] = stds[c]
        out[f"{c}_slope_{cfg.long_window}"] = slopes[c]
        out[f"{c}_delta_base"] = means[c] - base_aligned[c]

    informative = [c for c in cfg.informative if c in sensors]
    if informative:
        scaled = [WEAR_DIRECTION[c] * (means[c] - base_aligned[c]) / spread_aligned[c] for c in informative]
        out["health_indicator"] = np.mean(np.vstack([s.to_numpy() for s in scaled]), axis=0)
        hi = out.groupby(UNIT_COL, sort=False)["health_indicator"]
        for lag in cfg.hi_lags:
            out[f"health_indicator_diff_{lag}"] = out["health_indicator"] - hi.shift(lag).fillna(
                out["health_indicator"]
            )

    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL].clip(upper=RUL_CAP)
    return out


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and the target."""
    return [c for c in features.columns if c not in (UNIT_COL, TARGET_COL)]


def last_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent row of every unit (what a scoring run predicts on)."""
    idx = frame.groupby(UNIT_COL)[CYCLE_COL].idxmax()
    latest: pd.DataFrame = frame.loc[idx].reset_index(drop=True)
    return latest
