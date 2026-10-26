"""Batch scoring: the champion predicts the remaining life of every unit in service."""

from __future__ import annotations

import mlflow
import pandas as pd

from pdm.data import TARGET_COL
from pdm.features import last_cycle
from pdm.model import PREDICTION_COL

MAINTENANCE_FLAG = "maintenance_due"


def flag_maintenance(predictions: pd.DataFrame, threshold_cycles: float) -> pd.DataFrame:
    """Keep each unit's latest prediction and flag the ones below the planning threshold."""
    latest = last_cycle(predictions)
    latest[MAINTENANCE_FLAG] = latest[PREDICTION_COL] <= threshold_cycles
    ordered: pd.DataFrame = latest.sort_values(PREDICTION_COL).reset_index(drop=True)
    return ordered


def score_batch(model_uri: str, telemetry: pd.DataFrame, threshold_cycles: float = 30.0) -> pd.DataFrame:
    model = mlflow.pyfunc.load_model(model_uri)
    inputs = telemetry.drop(columns=[TARGET_COL], errors="ignore")
    return flag_maintenance(model.predict(inputs), threshold_cycles)
