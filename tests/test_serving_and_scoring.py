import importlib.util
import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from pdm.config import CHAMPION, Settings
from pdm.registry import register_and_gate
from pdm.scoring import MAINTENANCE_FLAG, flag_maintenance, score_batch
from pdm.train import TrainParams, train

REPO = Path(__file__).resolve().parents[1]


def test_flag_maintenance_keeps_latest_row_per_unit() -> None:
    preds = pd.DataFrame(
        {"unit_id": [1, 1, 2, 2], "cycle": [1, 2, 1, 2], "predicted_rul": [50.0, 25.0, 90.0, 80.0]}
    )
    out = flag_maintenance(preds, threshold_cycles=30)
    assert out["unit_id"].tolist() == [1, 2]
    assert out[MAINTENANCE_FLAG].tolist() == [True, False]


@pytest.fixture()
def champion_uri(
    mlflow_settings: Settings,
    small_features: pd.DataFrame,
    small_fleet: pd.DataFrame,
    small_holdout: pd.DataFrame,
) -> str:
    result = train(small_features, small_fleet, small_holdout, TrainParams(model_type="ridge"))
    register_and_gate(mlflow_settings, result.model_uri, small_holdout)
    return f"models:/{mlflow_settings.model_name}@{CHAMPION}"


def test_batch_scoring_with_champion_alias(champion_uri: str, small_holdout: pd.DataFrame) -> None:
    scored = score_batch(champion_uri, small_holdout.drop(columns=["rul"]), threshold_cycles=30)
    assert len(scored) == small_holdout["unit_id"].nunique()
    assert scored["predicted_rul"].is_monotonic_increasing


def test_azureml_score_script(
    champion_uri: str, small_holdout: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "azureml-model"
    mlflow.artifacts.download_artifacts(champion_uri, dst_path=str(model_dir / "pdm-rul"))
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MAINTENANCE_THRESHOLD_CYCLES", "200")

    spec = importlib.util.spec_from_file_location("score", REPO / "serving" / "azureml" / "score.py")
    assert spec and spec.loader
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    score.init()

    window = small_holdout[small_holdout["unit_id"].isin([1, 2])].drop(columns=["rul"])
    body = json.dumps({"data": json.loads(window.to_json(orient="records"))})
    out = score.run(body)["predictions"]
    assert [p["unit_id"] for p in out] == [1, 2]
    assert all(p["maintenance_due"] for p in out)
    assert score.run(json.dumps({"data": []})) == {"predictions": []}


def test_sample_request_matches_model_contract(champion_uri: str) -> None:
    sample = json.loads((REPO / "serving" / "azureml" / "sample-request.json").read_text())
    info = mlflow.models.get_model_info(champion_uri)
    assert set(sample["data"][0]) == {c.name for c in info.signature.inputs}
