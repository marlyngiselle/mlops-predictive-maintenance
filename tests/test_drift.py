from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.drift import ALERT, OK, WARN, Thresholds, compare, drift_report, psi

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def test_psi_of_identical_distributions_is_near_zero() -> None:
    x = RNG.normal(0, 1, 20_000)
    assert psi(x, RNG.normal(0, 1, 20_000)) < 0.01


def test_psi_grows_with_shift() -> None:
    ref = RNG.normal(0, 1, 20_000)
    small = psi(ref, RNG.normal(0.2, 1, 20_000))
    large = psi(ref, RNG.normal(1.0, 1, 20_000))
    assert 0.01 < small < 0.1
    assert large > 0.25


def test_psi_handles_constant_reference() -> None:
    assert psi(np.ones(100), np.ones(50)) == 0.0
    assert psi(np.ones(100), np.zeros(50)) == float("inf")


def test_psi_rejects_empty() -> None:
    with pytest.raises(ValueError):
        psi([], [1.0])


def test_classification_levels() -> None:
    t = Thresholds()
    ref = RNG.normal(0, 1, 5_000)
    assert compare("x", ref, RNG.normal(0, 1, 5_000), t).status == OK
    assert compare("x", ref, RNG.normal(0.4, 1, 5_000), t).status == WARN
    assert compare("x", ref, RNG.normal(1.5, 1, 5_000), t).status == ALERT


def test_tiny_ks_pvalue_alone_does_not_alert() -> None:
    """With large samples a negligible shift has a tiny p-value; the statistic gate stops the alert."""
    ref = RNG.normal(0, 1, 200_000)
    result = compare("x", ref, RNG.normal(0.05, 1, 200_000), Thresholds())
    assert result.ks_pvalue < 0.01
    assert result.status == OK


def test_report_flags_only_the_shifted_sensor() -> None:
    ref = pd.DataFrame({"a": RNG.normal(0, 1, 5_000), "b": RNG.normal(5, 2, 5_000)})
    cur = pd.DataFrame({"a": RNG.normal(0, 1, 5_000), "b": RNG.normal(9, 2, 5_000)})
    report = drift_report(ref, cur, ["a", "b"], reference_predictions=ref["a"], current_predictions=cur["a"])
    assert report.status == ALERT
    assert [a.split(":")[0] for a in report.alerts] == ["b"]
    payload = report.to_dict()
    assert payload["prediction"]["status"] == OK
    assert payload["reference_rows"] == 5_000


def test_prediction_drift_uses_its_own_thresholds() -> None:
    t = Thresholds(prediction_psi_warn=5.0, prediction_psi_alert=10.0, ks_stat_alert=1.1)
    ref = RNG.normal(0, 1, 5_000)
    assert compare("prediction", ref, RNG.normal(1.5, 1, 5_000), t, prediction=True).status == OK


def test_thresholds_file_matches_dataclass() -> None:
    t = Thresholds.from_yaml(REPO / "monitoring" / "thresholds.yml")
    assert t.psi_warn < t.psi_alert
    assert t == Thresholds()


def test_unknown_threshold_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "t.yml"
    bad.write_text("psi_alarm: 0.3\n")
    with pytest.raises(ValueError, match="psi_alarm"):
        Thresholds.from_yaml(bad)
