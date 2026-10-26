"""The deployable model: raw telemetry in, remaining useful life out.

Feature engineering travels inside the model, so batch scoring, the online endpoint
and the training run all use the same code path. There is no second implementation
of the features to drift away from the first one.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModel

from pdm.data import CYCLE_COL, UNIT_COL
from pdm.features import RUL_CAP, FeatureConfig, build_features

PREDICTION_COL = "predicted_rul"


def save_feature_config(config: FeatureConfig, path: str) -> None:
    payload = asdict(config)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_feature_config(path: str) -> FeatureConfig:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return FeatureConfig(
        sensors=tuple(payload["sensors"]),
        short_window=int(payload["short_window"]),
        long_window=int(payload["long_window"]),
        baseline_cycles=int(payload["baseline_cycles"]),
        hi_lags=tuple(int(x) for x in payload["hi_lags"]),
        informative=tuple(payload["informative"]),
    )


def predict_frame(
    estimator: Any, columns: list[str], config: FeatureConfig, raw: pd.DataFrame
) -> pd.DataFrame:
    feats = build_features(raw, config)
    preds = np.clip(estimator.predict(feats[columns]), 0.0, float(RUL_CAP))
    return pd.DataFrame(
        {UNIT_COL: feats[UNIT_COL].to_numpy(), CYCLE_COL: feats[CYCLE_COL].to_numpy(), PREDICTION_COL: preds}
    )


class RulModel(PythonModel):
    """MLflow pyfunc wrapper, logged as code (no pickled class) with its artifacts."""

    def load_context(self, context: Any) -> None:
        bundle = joblib.load(context.artifacts["estimator"])
        self._estimator = bundle["estimator"]
        self._columns: list[str] = bundle["columns"]
        self._config = load_feature_config(context.artifacts["feature_config"])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        return predict_frame(self._estimator, self._columns, self._config, model_input)
