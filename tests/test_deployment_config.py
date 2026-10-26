"""Static checks on the bundle and serving YAML (no workspace needed)."""

from pathlib import Path

import pytest
import yaml

from pdm.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "pipelines"


def load(path: Path) -> dict:  # type: ignore[type-arg]
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def jobs() -> dict:  # type: ignore[type-arg]
    merged: dict = {}  # type: ignore[type-arg]
    for f in sorted((PIPELINES / "resources").glob("*.yml")):
        merged.update(load(f)["resources"]["jobs"])
    return merged


def test_bundle_targets_and_variables() -> None:
    bundle = load(PIPELINES / "databricks.yml")
    assert set(bundle["targets"]) == {"dev", "prod"}
    assert bundle["targets"]["dev"]["mode"] == "development"
    assert bundle["targets"]["prod"]["mode"] == "production"
    assert "service_principal_name" in bundle["targets"]["prod"]["run_as"]
    assert (
        bundle["targets"]["dev"]["variables"]["catalog"] != bundle["targets"]["prod"]["variables"]["catalog"]
    )


def test_expected_jobs(jobs: dict) -> None:  # type: ignore[type-arg]
    assert set(jobs) == {
        "pdm_training",
        "pdm_batch_scoring",
        "pdm_drift_monitor",
        "pdm_bootstrap_synthetic_data",
    }


def test_training_task_order(jobs: dict) -> None:  # type: ignore[type-arg]
    tasks = {t["task_key"]: t for t in jobs["pdm_training"]["tasks"]}
    assert list(tasks) == ["feature_build", "train", "evaluate_and_register"]
    assert tasks["train"]["depends_on"] == [{"task_key": "feature_build"}]
    assert tasks["evaluate_and_register"]["depends_on"] == [{"task_key": "train"}]


def test_tasks_use_unity_catalog_and_known_subcommands(jobs: dict) -> None:  # type: ignore[type-arg]
    commands = set(build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    for job in jobs.values():
        for task in job["tasks"]:
            params = task["python_wheel_task"]["parameters"]
            assert task["python_wheel_task"]["entry_point"] == "pdm"
            assert any(p in commands for p in params)
            if "--registry-uri" in params:
                assert params[params.index("--registry-uri") + 1] == "databricks-uc"


def test_drift_job_fails_on_alert(jobs: dict) -> None:  # type: ignore[type-arg]
    params = jobs["pdm_drift_monitor"]["tasks"][0]["python_wheel_task"]["parameters"]
    assert "--fail-on-alert" in params


def test_azureml_yaml_parses() -> None:
    endpoint = load(REPO / "serving" / "azureml" / "endpoint.yml")
    deployment = load(REPO / "serving" / "azureml" / "deployment.yml")
    assert deployment["endpoint_name"] == endpoint["name"]
    assert deployment["code_configuration"]["scoring_script"] == "score.py"
    assert (REPO / "serving" / "azureml" / deployment["environment"]["conda_file"]).exists()


def test_databricks_serving_yaml_parses() -> None:
    cfg = load(REPO / "serving" / "databricks" / "model-serving.yml")
    assert "pdm_rul" in cfg["resources"]["model_serving_endpoints"]
