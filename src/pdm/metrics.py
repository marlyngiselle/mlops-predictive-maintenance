"""Evaluation metrics for remaining useful life.

RMSE and MAE treat early and late predictions the same. Maintenance does not: a
prediction that says a machine has 40 cycles left when it has 10 leads to an
unplanned stop. The asymmetric score below, from the prognostics literature,
grows faster for late predictions (time constant 10) than for early ones (13).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

EARLY_TAU = 13.0
LATE_TAU = 10.0


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(p - t)))


def asymmetric_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Sum of exp(-d/13) - 1 for early and exp(d/10) - 1 for late predictions, d = pred - true.

    Lower is better and a perfect model scores 0.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(d < 0, np.exp(-d / EARLY_TAU) - 1.0, np.exp(d / LATE_TAU) - 1.0)
    return float(np.sum(penalties))


def late_ratio(y_true: ArrayLike, y_pred: ArrayLike, tolerance: float = 10.0) -> float:
    """Share of predictions that overestimate the remaining life by more than ``tolerance`` cycles."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(d > tolerance)) if d.size else 0.0


@dataclass(frozen=True)
class Metrics:
    """Everything the promotion gate looks at, for one model on one dataset."""

    rmse: float
    mae: float
    score: float
    late_ratio: float
    n: int

    @classmethod
    def compute(cls, y_true: ArrayLike, y_pred: ArrayLike) -> Metrics:
        t = np.asarray(y_true, dtype=float)
        return cls(
            rmse=rmse(t, y_pred),
            mae=mae(t, y_pred),
            score=asymmetric_score(t, y_pred),
            late_ratio=late_ratio(t, y_pred),
            n=int(t.size),
        )

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": float(v) for k, v in asdict(self).items()}
