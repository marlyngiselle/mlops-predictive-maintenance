import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pdm.config import CHALLENGER, CHAMPION, PREVIOUS_CHAMPION, Settings
from pdm.metrics import Metrics
from pdm.registry import GatePolicy, decide, register_and_gate, rollback
from pdm.train import TrainParams, train


def m(rmse: float, score: float = 100.0) -> Metrics:
    return Metrics(rmse=rmse, mae=rmse * 0.8, score=score, late_ratio=0.1, n=100)


class TestDecide:
    def test_first_model_promoted_when_under_ceiling(self) -> None:
        d = decide(m(20.0), None, GatePolicy(max_rmse=30.0))
        assert d.promote
        assert "no current champion" in d.reasons

    def test_first_model_rejected_above_ceiling(self) -> None:
        assert not decide(m(31.0), None, GatePolicy(max_rmse=30.0)).promote

    def test_clear_improvement_promoted(self) -> None:
        assert decide(m(15.0, 90.0), m(16.5, 100.0)).promote

    def test_marginal_improvement_rejected(self) -> None:
        d = decide(m(15.9), m(16.0), GatePolicy(min_relative_improvement=0.02))
        assert not d.promote
        assert any("does not beat champion" in r for r in d.reasons)

    def test_better_rmse_but_more_late_predictions_rejected(self) -> None:
        d = decide(m(14.0, score=130.0), m(16.0, score=100.0))
        assert not d.promote
        assert any("late-penalising score" in r for r in d.reasons)

    def test_score_regression_tolerance(self) -> None:
        policy = GatePolicy(max_score_regression=0.5)
        assert decide(m(14.0, score=130.0), m(16.0, score=100.0), policy).promote

    def test_reasons_list_failures_only_when_rejected(self) -> None:
        d = decide(m(40.0, 500.0), m(16.0, 100.0))
        assert len(d.reasons) == 3
        assert d.summary.startswith("reject: ")


def _train(
    settings: Settings, feats: pd.DataFrame, raw: pd.DataFrame, holdout: pd.DataFrame, **kw: object
) -> str:
    return train(feats, raw, holdout, TrainParams(**kw), run_name=str(kw)).model_uri  # type: ignore[arg-type]


def test_gate_moves_aliases_end_to_end(
    mlflow_settings: Settings,
    small_features: pd.DataFrame,
    small_fleet: pd.DataFrame,
    small_holdout: pd.DataFrame,
) -> None:
    client = MlflowClient()
    name = mlflow_settings.model_name

    first = register_and_gate(
        mlflow_settings,
        _train(mlflow_settings, small_features, small_fleet, small_holdout, model_type="ridge"),
        small_holdout,
    )
    assert first.decision.promote
    assert str(client.get_model_version_by_alias(name, CHAMPION).version) == first.version

    weak = register_and_gate(
        mlflow_settings,
        _train(
            mlflow_settings,
            small_features,
            small_fleet,
            small_holdout,
            model_type="hgb",
            max_iter=3,
            max_leaf_nodes=2,
        ),
        small_holdout,
    )
    assert not weak.decision.promote
    assert weak.champion_version == first.version
    assert str(client.get_model_version_by_alias(name, CHAMPION).version) == first.version
    assert str(client.get_model_version_by_alias(name, CHALLENGER).version) == weak.version
    tags = client.get_model_version(name, weak.version).tags
    assert tags["gate_decision"] == "rejected"
    assert tags["gate_compared_with"] == first.version

    forced = register_and_gate(
        mlflow_settings,
        _train(
            mlflow_settings, small_features, small_fleet, small_holdout, model_type="ridge", ridge_alpha=5.0
        ),
        small_holdout,
        GatePolicy(min_relative_improvement=-1.0, max_score_regression=10.0),
    )
    assert forced.decision.promote
    assert str(client.get_model_version_by_alias(name, PREVIOUS_CHAMPION).version) == first.version

    assert rollback(mlflow_settings) == first.version
    assert str(client.get_model_version_by_alias(name, CHAMPION).version) == first.version
    assert str(client.get_model_version_by_alias(name, PREVIOUS_CHAMPION).version) == forced.version


def test_logged_model_has_signature_without_label(
    mlflow_settings: Settings,
    small_features: pd.DataFrame,
    small_fleet: pd.DataFrame,
    small_holdout: pd.DataFrame,
) -> None:
    uri = _train(mlflow_settings, small_features, small_fleet, small_holdout, model_type="ridge")
    info = mlflow.models.get_model_info(uri)
    inputs = {c.name for c in info.signature.inputs}
    outputs = {c.name for c in info.signature.outputs}
    assert "rul" not in inputs and "sensor_01" in inputs
    assert outputs == {"unit_id", "cycle", "predicted_rul"}
    preds = mlflow.pyfunc.load_model(uri).predict(small_holdout.drop(columns=["rul"]))
    assert preds["predicted_rul"].between(0, 125).all()


def test_unity_catalog_needs_three_level_name() -> None:
    with pytest.raises(ValueError, match=r"catalog\.schema\.model"):
        Settings(tracking_uri="databricks", registry_uri="databricks-uc", model_name="rul").validate()
    Settings(
        tracking_uri="databricks", registry_uri="databricks-uc", model_name="ml.pdm.rul_model"
    ).validate()


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDM_TRACKING_URI", "databricks")
    monkeypatch.setenv("PDM_REGISTRY_URI", "databricks-uc")
    monkeypatch.setenv("PDM_MODEL_NAME", "ml_prod.pdm.rul_model")
    s = Settings.from_env()
    assert s.uses_unity_catalog
    assert s.model_name == "ml_prod.pdm.rul_model"
