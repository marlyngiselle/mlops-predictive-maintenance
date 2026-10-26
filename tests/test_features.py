import numpy as np
import pandas as pd

from pdm.data import CYCLE_COL, TARGET_COL, UNIT_COL
from pdm.features import RUL_CAP, FeatureConfig, build_features, feature_columns, last_cycle


def test_no_missing_values(small_features: pd.DataFrame) -> None:
    assert not small_features.isna().any().any()


def test_target_is_capped(small_features: pd.DataFrame) -> None:
    assert small_features[TARGET_COL].max() == RUL_CAP
    assert small_features[TARGET_COL].min() == 0


def test_feature_families_present(small_features: pd.DataFrame) -> None:
    cols = feature_columns(small_features)
    cfg = FeatureConfig()
    assert f"sensor_01_mean_{cfg.short_window}" in cols
    assert f"sensor_01_std_{cfg.short_window}" in cols
    assert f"sensor_01_slope_{cfg.long_window}" in cols
    assert "sensor_01_delta_base" in cols
    assert "health_indicator" in cols
    assert all(f"health_indicator_diff_{lag}" in cols for lag in cfg.hi_lags)
    assert TARGET_COL not in cols and UNIT_COL not in cols


def test_features_never_look_ahead(small_fleet: pd.DataFrame) -> None:
    """Features at cycle t must be identical whether or not later cycles exist."""
    unit = small_fleet[small_fleet[UNIT_COL] == 1]
    full = build_features(unit).set_index(CYCLE_COL)
    cutoff = 60
    partial = build_features(unit[unit[CYCLE_COL] <= cutoff]).set_index(CYCLE_COL)
    cols = feature_columns(partial.reset_index())
    cols.remove(CYCLE_COL)
    pd.testing.assert_frame_equal(full.loc[:cutoff, cols], partial[cols])


def test_features_ignore_other_units(small_fleet: pd.DataFrame) -> None:
    both = build_features(small_fleet[small_fleet[UNIT_COL].isin([1, 2])])
    alone = build_features(small_fleet[small_fleet[UNIT_COL] == 2])
    cols = feature_columns(alone)
    got = both[both[UNIT_COL] == 2][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, alone[cols].reset_index(drop=True))


def test_health_indicator_rises_with_wear(small_features: pd.DataFrame) -> None:
    corr = small_features.groupby(UNIT_COL).apply(
        lambda u: np.corrcoef(u[CYCLE_COL], u["health_indicator"])[0, 1]
    )
    assert corr.median() > 0.8


def test_slope_recovers_a_linear_trend() -> None:
    cycles = np.arange(1, 61)
    frame = pd.DataFrame({UNIT_COL: 1, CYCLE_COL: cycles})
    for i in range(1, 14):
        frame[f"sensor_{i:02d}"] = 2.0 * cycles + 5.0
    feats = build_features(frame)
    assert np.allclose(feats["sensor_01_slope_30"].iloc[30:], 2.0)


def test_last_cycle_picks_latest_row(small_features: pd.DataFrame) -> None:
    latest = last_cycle(small_features)
    assert len(latest) == small_features[UNIT_COL].nunique()
    expected = small_features.groupby(UNIT_COL)[CYCLE_COL].max().to_numpy()
    assert (latest.sort_values(UNIT_COL)[CYCLE_COL].to_numpy() == expected).all()
