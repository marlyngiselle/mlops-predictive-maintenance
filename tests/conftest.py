from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from pdm.config import Settings, configure_mlflow
from pdm.data import FleetConfig, generate_fleet, truncate_fleet
from pdm.features import build_features

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def small_fleet() -> pd.DataFrame:
    return generate_fleet(FleetConfig(n_units=24, seed=11))


@pytest.fixture(scope="session")
def small_holdout() -> pd.DataFrame:
    return truncate_fleet(generate_fleet(FleetConfig(n_units=30, seed=12)), seed=13)


@pytest.fixture(scope="session")
def small_features(small_fleet: pd.DataFrame) -> pd.DataFrame:
    return build_features(small_fleet)


@pytest.fixture()
def mlflow_settings(tmp_path: Path) -> Iterator[Settings]:
    """An isolated MLflow backend (SQLite file plus local artifacts) per test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("pdm-test", artifact_location=(tmp_path / "mlruns").as_uri())
    settings = Settings(tracking_uri=uri, experiment="pdm-test", model_name="pdm_rul_test")
    configure_mlflow(settings)
    yield settings
    if mlflow.active_run():
        mlflow.end_run()
