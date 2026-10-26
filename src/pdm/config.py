"""Where MLflow writes and which registry it uses.

The same code runs in three places, and only the environment changes:

| Where              | PDM_TRACKING_URI      | PDM_REGISTRY_URI      | PDM_MODEL_NAME            |
| ------------------ | --------------------- | --------------------- | ------------------------- |
| Laptop and tests   | sqlite:///mlflow.db   | (same as tracking)    | pdm_rul                   |
| Databricks dev     | databricks            | databricks-uc         | ml_dev.pdm.rul_model      |
| Databricks prod    | databricks            | databricks-uc         | ml_prod.pdm.rul_model     |

Unity Catalog needs a three level model name and a model signature. Both are enforced
here and in the training code, so a run that would fail registration fails early.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlflow

UC_REGISTRY = "databricks-uc"
CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS_CHAMPION = "previous_champion"


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = "sqlite:///mlflow.db"
    registry_uri: str | None = None
    experiment: str = "pdm-rul"
    model_name: str = "pdm_rul"

    @property
    def effective_registry_uri(self) -> str:
        return self.registry_uri or self.tracking_uri

    @property
    def uses_unity_catalog(self) -> bool:
        return self.effective_registry_uri == UC_REGISTRY

    def validate(self) -> None:
        if self.uses_unity_catalog and self.model_name.count(".") != 2:
            raise ValueError(f"Unity Catalog model names are catalog.schema.model, got {self.model_name!r}")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            tracking_uri=os.environ.get("PDM_TRACKING_URI", defaults.tracking_uri),
            registry_uri=os.environ.get("PDM_REGISTRY_URI") or None,
            experiment=os.environ.get("PDM_EXPERIMENT", defaults.experiment),
            model_name=os.environ.get("PDM_MODEL_NAME", defaults.model_name),
        )


def configure_mlflow(settings: Settings) -> None:
    """Point the MLflow client at the tracking server, the registry and the experiment."""
    settings.validate()
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_registry_uri(settings.effective_registry_uri)
    mlflow.set_experiment(settings.experiment)
