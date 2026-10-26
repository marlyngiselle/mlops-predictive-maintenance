"""Entry point for MLflow models from code. MLflow executes this file when it loads the model."""

import mlflow

from pdm.model import RulModel

mlflow.models.set_model(RulModel())
