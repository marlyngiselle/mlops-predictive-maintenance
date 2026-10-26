import pandas as pd
import pytest

from pdm.data import (
    CYCLE_COL,
    SENSOR_COLS,
    TARGET_COL,
    UNIT_COL,
    FleetConfig,
    apply_sensor_shift,
    generate_fleet,
    truncate_fleet,
)


def test_same_seed_same_fleet() -> None:
    a = generate_fleet(FleetConfig(n_units=5, seed=3))
    b = generate_fleet(FleetConfig(n_units=5, seed=3))
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_different_fleet() -> None:
    a = generate_fleet(FleetConfig(n_units=5, seed=3))
    b = generate_fleet(FleetConfig(n_units=5, seed=4))
    assert not a[list(SENSOR_COLS)].equals(b[list(SENSOR_COLS)])


def test_every_unit_runs_to_failure_within_bounds() -> None:
    cfg = FleetConfig(n_units=20, min_life=100, max_life=150, seed=1)
    fleet = generate_fleet(cfg)
    lives = fleet.groupby(UNIT_COL).size()
    assert lives.between(cfg.min_life, cfg.max_life).all()
    last = fleet.groupby(UNIT_COL).tail(1)
    assert (last[TARGET_COL] == 0).all()
    assert fleet.groupby(UNIT_COL)[TARGET_COL].apply(lambda s: s.is_monotonic_decreasing).all()


def test_informative_sensor_moves_with_wear() -> None:
    fleet = generate_fleet(FleetConfig(n_units=30, seed=5))
    # sensor_02 rises with damage, sensor_04 falls, sensor_11 carries no signal.
    assert fleet["sensor_02"].corr(-fleet[TARGET_COL]) > 0.5
    assert fleet["sensor_04"].corr(-fleet[TARGET_COL]) < -0.5
    assert abs(fleet["sensor_11"].corr(-fleet[TARGET_COL])) < 0.05


def test_truncation_keeps_prefix_and_positive_rul() -> None:
    fleet = generate_fleet(FleetConfig(n_units=10, seed=2))
    cut = truncate_fleet(fleet, seed=9)
    assert set(cut[UNIT_COL]) == set(fleet[UNIT_COL])
    assert (cut.groupby(UNIT_COL)[TARGET_COL].min() > 0).all()
    assert (cut.groupby(UNIT_COL)[CYCLE_COL].min() == 1).all()
    pd.testing.assert_frame_equal(truncate_fleet(fleet, seed=9), cut)


def test_sensor_shift_is_a_copy() -> None:
    fleet = generate_fleet(FleetConfig(n_units=2, seed=2))
    shifted = apply_sensor_shift(fleet, "sensor_04", 3.0)
    assert (shifted["sensor_04"] - fleet["sensor_04"]).round(9).eq(3.0).all()
    assert shifted is not fleet


@pytest.mark.parametrize("n_units", [1, 7])
def test_unit_ids_are_contiguous(n_units: int) -> None:
    fleet = generate_fleet(FleetConfig(n_units=n_units, seed=0))
    assert sorted(fleet[UNIT_COL].unique()) == list(range(1, n_units + 1))
