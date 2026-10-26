"""Scoring script for the Azure ML managed online endpoint.

The deployment mounts the champion MLflow model under ``AZUREML_MODEL_DIR``. The model
carries its own feature engineering, so this script only parses the request, calls the
model and shapes the answer for the maintenance planner.

Request body::

    {"data": [{"unit_id": 7, "cycle": 1, "op_setting_1": 0.0, ..., "sensor_13": 0.03}, ...]}

Send at least the last 30 cycles per unit so the rolling features have a full window.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

LOG = logging.getLogger("pdm.score")
_MODEL: Any = None
_THRESHOLD = 30.0


def _model_path(root: str) -> str:
    """Return the folder holding ``MLmodel`` (Azure ML adds one level for the model name)."""
    candidates = sorted(Path(root).rglob("MLmodel"))
    if not candidates:
        raise FileNotFoundError(f"no MLmodel file under {root}")
    return str(candidates[0].parent)


def init() -> None:
    global _MODEL, _THRESHOLD  # noqa: PLW0603 (Azure ML calls init once per worker)
    _MODEL = mlflow.pyfunc.load_model(_model_path(os.environ["AZUREML_MODEL_DIR"]))
    _THRESHOLD = float(os.environ.get("MAINTENANCE_THRESHOLD_CYCLES", "30"))
    LOG.info("model loaded, maintenance threshold %s cycles", _THRESHOLD)


def run(raw_data: str) -> dict[str, Any]:
    payload = json.loads(raw_data)
    telemetry = pd.DataFrame(payload["data"])
    if telemetry.empty:
        return {"predictions": []}
    predictions = _MODEL.predict(telemetry)
    latest = predictions.loc[predictions.groupby("unit_id")["cycle"].idxmax()]
    return {
        "predictions": [
            {
                "unit_id": int(row.unit_id),
                "cycle": int(row.cycle),
                "predicted_rul": round(float(row.predicted_rul), 1),
                "maintenance_due": bool(row.predicted_rul <= _THRESHOLD),
            }
            for row in latest.itertuples(index=False)
        ]
    }
