"""Data drift and prediction drift.

Two complementary tests per feature:

* PSI (population stability index) on reference quantile bins. Robust to sample
  size, which makes it a good alerting signal. 0.1 is the usual warning level and
  0.25 the usual action level.
* Two sample Kolmogorov Smirnov. Sensitive to any change in shape, but with tens of
  thousands of rows the p-value is tiny for differences nobody cares about, so it
  only alerts when the statistic itself is also large.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp

OK, WARN, ALERT = "ok", "warn", "alert"
_SEVERITY = {OK: 0, WARN: 1, ALERT: 2}
_EPS = 1e-4


def psi(reference: ArrayLike, current: ArrayLike, bins: int = 10) -> float:
    """Population stability index of ``current`` against ``reference`` on reference quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("psi needs non-empty samples")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / ref.size
    cur_share = np.histogram(cur, edges)[0] / cur.size
    ref_share, cur_share = np.clip(ref_share, _EPS, None), np.clip(cur_share, _EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


@dataclass(frozen=True)
class Thresholds:
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_pvalue: float = 0.01
    ks_stat_alert: float = 0.20
    prediction_psi_warn: float = 0.10
    prediction_psi_alert: float = 0.25
    bins: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> Thresholds:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class DriftResult:
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    status: str


def _classify(psi_value: float, ks_stat: float, ks_p: float, warn: float, alert: float, t: Thresholds) -> str:
    if psi_value >= alert or (ks_p < t.ks_pvalue and ks_stat >= t.ks_stat_alert):
        return ALERT
    if psi_value >= warn:
        return WARN
    return OK


def compare(
    name: str, reference: ArrayLike, current: ArrayLike, t: Thresholds, prediction: bool = False
) -> DriftResult:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    p = psi(ref, cur, t.bins)
    ks = ks_2samp(ref, cur)
    warn, alert = (t.prediction_psi_warn, t.prediction_psi_alert) if prediction else (t.psi_warn, t.psi_alert)
    stat, pvalue = float(ks.statistic), float(ks.pvalue)
    return DriftResult(name, round(p, 4), round(stat, 4), pvalue, _classify(p, stat, pvalue, warn, alert, t))


@dataclass
class DriftReport:
    features: list[DriftResult]
    prediction: DriftResult | None = None
    reference_rows: int = 0
    current_rows: int = 0
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def status(self) -> str:
        results = [*self.features, *([self.prediction] if self.prediction else [])]
        return max((r.status for r in results), key=_SEVERITY.__getitem__, default=OK)

    @property
    def alerts(self) -> list[str]:
        results = [*self.features, *([self.prediction] if self.prediction else [])]
        return [f"{r.name}: psi={r.psi:.3f} ks={r.ks_stat:.3f}" for r in results if r.status == ALERT]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "alerts": self.alerts,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "thresholds": asdict(self.thresholds),
            "prediction": asdict(self.prediction) if self.prediction else None,
            "features": [asdict(r) for r in self.features],
        }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str],
    thresholds: Thresholds | None = None,
    reference_predictions: ArrayLike | None = None,
    current_predictions: ArrayLike | None = None,
) -> DriftReport:
    t = thresholds or Thresholds()
    results = [compare(f, reference[f], current[f], t) for f in features]
    prediction = None
    if reference_predictions is not None and current_predictions is not None:
        prediction = compare("prediction", reference_predictions, current_predictions, t, prediction=True)
    return DriftReport(results, prediction, len(reference), len(current), t)
