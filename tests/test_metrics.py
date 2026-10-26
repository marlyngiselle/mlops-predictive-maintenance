import math

import pytest

from pdm.metrics import Metrics, asymmetric_score, late_ratio, mae, rmse


def test_perfect_prediction_scores_zero() -> None:
    y = [10, 50, 100]
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert asymmetric_score(y, y) == 0.0


def test_late_is_penalised_more_than_early() -> None:
    early = asymmetric_score([50], [40])  # predicts failure 10 cycles too soon
    late = asymmetric_score([50], [60])  # predicts failure 10 cycles too late
    assert late > early
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.e - 1)


def test_rmse_and_mae_known_values() -> None:
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert mae([0, 0], [3, -4]) == pytest.approx(3.5)


def test_late_ratio_uses_tolerance() -> None:
    assert late_ratio([10, 10, 10, 10], [10, 21, 19, 0]) == 0.25
    assert late_ratio([], []) == 0.0


def test_metrics_bundle() -> None:
    m = Metrics.compute([10, 20], [12, 18])
    assert m.n == 2
    assert m.rmse == pytest.approx(2.0)
    assert set(m.as_dict("holdout_")) == {
        "holdout_rmse",
        "holdout_mae",
        "holdout_score",
        "holdout_late_ratio",
        "holdout_n",
    }
