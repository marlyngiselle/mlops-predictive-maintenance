"""Training with MLflow tracking, a model signature and lineage tags."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pdm
from pdm.data import CYCLE_COL, TARGET_COL, UNIT_COL
from pdm.features import FeatureConfig, build_features, feature_columns, last_cycle
from pdm.metrics import Metrics
from pdm.model import PREDICTION_COL, predict_frame, save_feature_config

MODEL_FILE = Path(pdm.__file__).with_name("mlflow_model.py")
PACKAGE_DIR = Path(pdm.__file__).parent


@dataclass(frozen=True)
class TrainParams:
    """Hyperparameters. ``model_type`` is ``hgb`` (gradient boosting) or ``ridge`` (linear baseline)."""

    model_type: str = "hgb"
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    ridge_alpha: float = 1.0
    validation_fraction: float = 0.2
    seed: int = 7


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    model_uri: str
    validation: Metrics
    holdout: Metrics


def make_estimator(params: TrainParams) -> Any:
    if params.model_type == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=params.max_iter,
            learning_rate=params.learning_rate,
            max_leaf_nodes=params.max_leaf_nodes,
            min_samples_leaf=params.min_samples_leaf,
            l2_regularization=params.l2_regularization,
            random_state=params.seed,
        )
    if params.model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params.ridge_alpha))
    raise ValueError(f"unknown model_type {params.model_type!r}")


def evaluate_on_holdout(predictions: pd.DataFrame, holdout_raw: pd.DataFrame) -> Metrics:
    """Compare the prediction at each unit's last observed cycle with its capped true RUL."""
    truth = last_cycle(build_features(holdout_raw))[[UNIT_COL, CYCLE_COL, TARGET_COL]]
    joined = truth.merge(last_cycle(predictions), on=[UNIT_COL, CYCLE_COL], how="inner")
    return Metrics.compute(joined[TARGET_COL], joined[PREDICTION_COL])


def _pinned_requirements() -> list[str]:
    pins = []
    for dist in ("numpy", "pandas", "scikit-learn", "mlflow-skinny"):
        try:
            pins.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pins.append(dist)
    return pins


def _git_sha() -> str:
    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("PDM_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def train(
    features: pd.DataFrame,
    raw_sample: pd.DataFrame,
    holdout_raw: pd.DataFrame,
    params: TrainParams | None = None,
    feature_config: FeatureConfig | None = None,
    run_name: str | None = None,
    data_version: str = "synthetic-v1",
) -> TrainResult:
    """Fit on whole units (no unit appears in both splits), log everything, return the model URI.

    ``mlflow`` must already point at the right tracking server and experiment
    (see :func:`pdm.config.configure_mlflow`).
    """
    p = params or TrainParams()
    cfg = feature_config or FeatureConfig()
    cols = feature_columns(features)

    splitter = GroupShuffleSplit(n_splits=1, test_size=p.validation_fraction, random_state=p.seed)
    train_idx, val_idx = next(splitter.split(features, groups=features[UNIT_COL]))
    tr, va = features.iloc[train_idx], features.iloc[val_idx]

    estimator = make_estimator(p)
    estimator.fit(tr[cols], tr[TARGET_COL])
    validation = Metrics.compute(va[TARGET_COL], np.clip(estimator.predict(va[cols]), 0, None))
    holdout = evaluate_on_holdout(predict_frame(estimator, cols, cfg, holdout_raw), holdout_raw)

    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags(
            {
                "pdm.git_sha": _git_sha(),
                "pdm.data_version": data_version,
                "pdm.package_version": pdm.__version__,
                "pdm.model_type": p.model_type,
            }
        )
        mlflow.log_params({f"hp_{k}": v for k, v in asdict(p).items()})
        mlflow.log_params(cfg.as_params())
        mlflow.log_params(
            {
                "n_features": len(cols),
                "n_train_units": tr[UNIT_COL].nunique(),
                "n_val_units": va[UNIT_COL].nunique(),
                "n_train_rows": len(tr),
            }
        )
        mlflow.log_metrics(validation.as_dict("val_"))
        mlflow.log_metrics(holdout.as_dict("holdout_"))

        estimator_path = os.path.join(tmp, "estimator.joblib")
        config_path = os.path.join(tmp, "feature_config.json")
        joblib.dump({"estimator": estimator, "columns": cols}, estimator_path)
        save_feature_config(cfg, config_path)

        # The serving contract is raw telemetry without the label.
        example = raw_sample.drop(columns=[TARGET_COL], errors="ignore").head(40).reset_index(drop=True)
        signature = infer_signature(example, predict_frame(estimator, cols, cfg, example))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(MODEL_FILE),
            artifacts={"estimator": estimator_path, "feature_config": config_path},
            code_paths=[str(PACKAGE_DIR)],
            signature=signature,
            input_example=example,
            pip_requirements=_pinned_requirements(),
        )
        return TrainResult(
            run_id=run.info.run_id, model_uri=info.model_uri, validation=validation, holdout=holdout
        )
