"""Registration and the champion/challenger gate.

Every trained model is registered and gets the ``challenger`` alias. It becomes
``champion`` only when it passes the gate against the current champion, scored on
the same holdout. The previous champion keeps the ``previous_champion`` alias, so a
rollback is one alias move and never a retrain.

The decision itself (:func:`decide`) is a pure function, so it is unit tested
without MLflow. The orchestration (:func:`register_and_gate`) works against any
registry: a local SQLite file in tests, Unity Catalog on Databricks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pdm.config import CHALLENGER, CHAMPION, PREVIOUS_CHAMPION, Settings
from pdm.data import TARGET_COL
from pdm.metrics import Metrics
from pdm.train import evaluate_on_holdout


@dataclass(frozen=True)
class GatePolicy:
    """What a challenger must prove before it serves traffic.

    * ``max_rmse``: absolute ceiling, also applied when there is no champion yet
    * ``min_relative_improvement``: RMSE must drop by at least this share of the champion's RMSE
    * ``max_score_regression``: the asymmetric (late-penalising) score may not grow by more than this share
    """

    max_rmse: float = 30.0
    min_relative_improvement: float = 0.02
    max_score_regression: float = 0.0


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        verdict = "promote" if self.promote else "reject"
        return f"{verdict}: " + "; ".join(self.reasons)


def decide(challenger: Metrics, champion: Metrics | None, policy: GatePolicy | None = None) -> Decision:
    """Return whether the challenger replaces the champion, with every reason spelled out."""
    pol = policy or GatePolicy()
    passed: list[str] = []
    failed: list[str] = []

    if challenger.rmse <= pol.max_rmse:
        passed.append(f"rmse {challenger.rmse:.2f} within ceiling {pol.max_rmse:.2f}")
    else:
        failed.append(f"rmse {challenger.rmse:.2f} above ceiling {pol.max_rmse:.2f}")

    if champion is None:
        passed.append("no current champion")
    else:
        needed = champion.rmse * (1.0 - pol.min_relative_improvement)
        if challenger.rmse <= needed:
            passed.append(
                f"rmse {challenger.rmse:.2f} beats champion {champion.rmse:.2f} (needed {needed:.2f})"
            )
        else:
            failed.append(
                f"rmse {challenger.rmse:.2f} does not beat champion {champion.rmse:.2f} (needed {needed:.2f})"
            )
        allowed = champion.score * (1.0 + pol.max_score_regression)
        if challenger.score <= allowed:
            passed.append(f"late-penalising score {challenger.score:.1f} <= {allowed:.1f}")
        else:
            failed.append(f"late-penalising score {challenger.score:.1f} > {allowed:.1f}")

    return Decision(promote=not failed, reasons=tuple(failed or passed))


@dataclass(frozen=True)
class GateOutcome:
    version: str
    decision: Decision
    challenger: Metrics
    champion: Metrics | None
    champion_version: str | None


def _champion_version(client: MlflowClient, name: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, CHAMPION).version)
    except MlflowException:
        return None


def score_model(model_uri: str, holdout_raw: pd.DataFrame) -> Metrics:
    model = mlflow.pyfunc.load_model(model_uri)
    telemetry = holdout_raw.drop(columns=[TARGET_COL], errors="ignore")
    return evaluate_on_holdout(model.predict(telemetry), holdout_raw)


def register_and_gate(
    settings: Settings,
    model_uri: str,
    holdout_raw: pd.DataFrame,
    policy: GatePolicy | None = None,
) -> GateOutcome:
    """Register ``model_uri``, score it and the champion on the same holdout, move aliases."""
    client = MlflowClient(registry_uri=settings.effective_registry_uri)
    name = settings.model_name

    version = str(mlflow.register_model(model_uri, name).version)
    client.set_registered_model_alias(name, CHALLENGER, version)

    challenger = score_model(f"models:/{name}/{version}", holdout_raw)
    champion_version = _champion_version(client, name)
    champion = score_model(f"models:/{name}@{CHAMPION}", holdout_raw) if champion_version else None

    decision = decide(challenger, champion, policy)
    for key, value in challenger.as_dict("gate_").items():
        client.set_model_version_tag(name, version, key, f"{value:.4f}")
    client.set_model_version_tag(
        name, version, "gate_decision", "promoted" if decision.promote else "rejected"
    )
    client.set_model_version_tag(name, version, "gate_reasons", " | ".join(decision.reasons))
    client.set_model_version_tag(name, version, "gate_compared_with", champion_version or "none")

    if decision.promote:
        if champion_version:
            client.set_registered_model_alias(name, PREVIOUS_CHAMPION, champion_version)
        client.set_registered_model_alias(name, CHAMPION, version)

    return GateOutcome(version, decision, challenger, champion, champion_version)


def rollback(settings: Settings) -> str:
    """Point ``champion`` back at ``previous_champion``. Returns the version now serving."""
    client = MlflowClient(registry_uri=settings.effective_registry_uri)
    name = settings.model_name
    previous = str(client.get_model_version_by_alias(name, PREVIOUS_CHAMPION).version)
    current = _champion_version(client, name)
    client.set_registered_model_alias(name, CHAMPION, previous)
    if current:
        client.set_registered_model_alias(name, PREVIOUS_CHAMPION, current)
    return previous
