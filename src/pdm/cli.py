"""Command line entry point. The Databricks jobs call these same subcommands from the wheel."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from pdm.config import CHAMPION, Settings, configure_mlflow
from pdm.data import (
    SENSOR_COLS,
    TARGET_COL,
    FleetConfig,
    apply_sensor_shift,
    generate_fleet,
    truncate_fleet,
)
from pdm.drift import ALERT, Thresholds, drift_report
from pdm.features import build_features, feature_columns
from pdm.model import PREDICTION_COL
from pdm.registry import GatePolicy, register_and_gate
from pdm.scoring import MAINTENANCE_FLAG, score_batch
from pdm.train import TrainParams, train

TRAIN_FILE, FEATURES_FILE = "train.parquet", "features.parquet"
# The holdout is the in-service snapshot the gate scores on. It is also the drift
# reference, since it is the population the champion was last validated against.
HOLDOUT_FILE = "holdout.parquet"
# A later in-service snapshot, drawn from the same process, that drift compares with the holdout.
CURRENT_FILE = "current.parquet"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _settings(a: argparse.Namespace) -> Settings:
    """Environment first, then explicit flags (the Databricks jobs pass flags)."""
    base = Settings.from_env()
    return Settings(
        tracking_uri=a.tracking_uri or base.tracking_uri,
        registry_uri=a.registry_uri or base.registry_uri,
        experiment=a.experiment or base.experiment,
        model_name=a.model_name or base.model_name,
    )


def cmd_generate(a: argparse.Namespace) -> int:
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_fleet(FleetConfig(n_units=a.units, seed=a.seed)).to_parquet(out / TRAIN_FILE, index=False)
    for name, offset in ((HOLDOUT_FILE, 1), (CURRENT_FILE, 3)):
        fleet = generate_fleet(FleetConfig(n_units=a.holdout_units, seed=a.seed + offset))
        truncate_fleet(fleet, a.seed + offset + 1).to_parquet(out / name, index=False)
    print(f"wrote {TRAIN_FILE}, {HOLDOUT_FILE} and {CURRENT_FILE} to {out}")
    return 0


def cmd_features(a: argparse.Namespace) -> int:
    feats = build_features(pd.read_parquet(Path(a.data_dir) / TRAIN_FILE))
    feats.to_parquet(Path(a.data_dir) / FEATURES_FILE, index=False)
    print(f"{len(feats)} rows, {feats.shape[1]} columns")
    return 0


def _params(a: argparse.Namespace) -> TrainParams:
    return TrainParams(
        model_type=a.model_type,
        max_iter=a.max_iter,
        learning_rate=a.learning_rate,
        max_leaf_nodes=a.max_leaf_nodes,
    )


def cmd_train(a: argparse.Namespace) -> int:
    configure_mlflow(_settings(a))
    data = Path(a.data_dir)
    result = train(
        pd.read_parquet(data / FEATURES_FILE),
        pd.read_parquet(data / TRAIN_FILE),
        pd.read_parquet(data / HOLDOUT_FILE),
        _params(a),
        run_name=a.run_name,
    )
    _write_json(Path(a.handoff), {"run_id": result.run_id, "model_uri": result.model_uri})
    print(json.dumps({"run_id": result.run_id, **result.holdout.as_dict("holdout_")}, indent=2))
    return 0


def cmd_evaluate_register(a: argparse.Namespace) -> int:
    settings = _settings(a)
    configure_mlflow(settings)
    handoff = json.loads(Path(a.handoff).read_text(encoding="utf-8"))
    policy = GatePolicy(a.max_rmse, a.min_improvement, a.max_score_regression)
    outcome = register_and_gate(
        settings, handoff["model_uri"], pd.read_parquet(Path(a.data_dir) / HOLDOUT_FILE), policy
    )
    print(f"version {outcome.version}: {outcome.decision.summary}")
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    settings = _settings(a)
    configure_mlflow(settings)
    uri = a.model_uri or f"models:/{settings.model_name}@{CHAMPION}"
    scored = score_batch(uri, pd.read_parquet(a.input), a.threshold)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(a.output, index=False)
    print(f"{len(scored)} units scored, {int(scored[MAINTENANCE_FLAG].sum())} due for maintenance")
    return 0


def _run_drift(reference: pd.DataFrame, current: pd.DataFrame, model_uri: str | None, t: Thresholds) -> Any:
    ref_pred = cur_pred = None
    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri)
        ref_pred = model.predict(reference.drop(columns=[TARGET_COL], errors="ignore"))[PREDICTION_COL]
        cur_pred = model.predict(current.drop(columns=[TARGET_COL], errors="ignore"))[PREDICTION_COL]
    return drift_report(reference, current, SENSOR_COLS, t, ref_pred, cur_pred)


def cmd_drift(a: argparse.Namespace) -> int:
    settings = _settings(a)
    configure_mlflow(settings)
    thresholds = Thresholds.from_yaml(a.thresholds) if a.thresholds else Thresholds()
    uri = None if a.no_predictions else (a.model_uri or f"models:/{settings.model_name}@{CHAMPION}")
    report = _run_drift(pd.read_parquet(a.reference), pd.read_parquet(a.current), uri, thresholds)
    _write_json(Path(a.output), report.to_dict())
    print(f"drift status: {report.status}")
    for line in report.alerts:
        print(f"ALERT {line}")
    return 2 if (a.fail_on_alert and report.status == ALERT) else 0


def cmd_demo(a: argparse.Namespace) -> int:
    """Run the whole lifecycle locally, against a SQLite MLflow backend, and write a summary."""
    work = Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = work / "data"
    started = time.perf_counter()
    cmd_generate(
        argparse.Namespace(output_dir=data, units=a.units, holdout_units=a.holdout_units, seed=a.seed)
    )
    cmd_features(argparse.Namespace(data_dir=data))

    settings = Settings(tracking_uri=f"sqlite:///{work / 'mlflow.db'}", experiment="pdm-rul-demo")
    mlflow.set_tracking_uri(settings.tracking_uri)
    if mlflow.get_experiment_by_name(settings.experiment) is None:
        mlflow.create_experiment(settings.experiment, artifact_location=(work / "mlruns").as_uri())
    configure_mlflow(settings)

    train_raw = pd.read_parquet(data / TRAIN_FILE)
    holdout = pd.read_parquet(data / HOLDOUT_FILE)
    feats = pd.read_parquet(data / FEATURES_FILE)
    candidates = [
        ("ridge-baseline", TrainParams(model_type="ridge")),
        ("hgb-300", TrainParams(model_type="hgb", max_iter=300)),
        (
            "hgb-500-slow",
            TrainParams(model_type="hgb", max_iter=500, learning_rate=0.03, min_samples_leaf=80),
        ),
        ("hgb-underfit", TrainParams(model_type="hgb", max_iter=8, learning_rate=0.05, max_leaf_nodes=4)),
    ]
    rounds = []
    for name, params in candidates:
        result = train(feats, train_raw, holdout, params, run_name=name)
        outcome = register_and_gate(settings, result.model_uri, holdout)
        rounds.append(
            {
                "candidate": name,
                "version": outcome.version,
                "validation": asdict(result.validation),
                "holdout": asdict(outcome.challenger),
                "champion_before": outcome.champion_version,
                "decision": outcome.decision.summary,
            }
        )
        print(f"{name:>15}  v{outcome.version}  {outcome.decision.summary}")

    champion_uri = f"models:/{settings.model_name}@{CHAMPION}"
    current = pd.read_parquet(data / CURRENT_FILE)
    scored = score_batch(champion_uri, current, a.threshold)
    t = Thresholds.from_yaml(a.thresholds) if a.thresholds else Thresholds()
    stable = _run_drift(holdout, current, champion_uri, t)
    shifted = _run_drift(holdout, apply_sensor_shift(current, "sensor_04", a.shift), champion_uri, t)
    _write_json(work / "drift_stable.json", stable.to_dict())
    _write_json(work / "drift_shifted.json", shifted.to_dict())

    summary = {
        "data": {
            "train_units": int(train_raw["unit_id"].nunique()),
            "train_rows": len(train_raw),
            "holdout_units": int(holdout["unit_id"].nunique()),
            "holdout_rows": len(holdout),
            "current_units": int(current["unit_id"].nunique()),
            "model_inputs": len(feature_columns(feats)),
        },
        "rounds": rounds,
        "batch_scoring": {
            "units": len(scored),
            "maintenance_due": int(scored[MAINTENANCE_FLAG].sum()),
            "threshold_cycles": a.threshold,
        },
        "drift": {
            "stable_status": stable.status,
            "stable_alerts": stable.alerts,
            "shifted_sensor": "sensor_04",
            "shift": a.shift,
            "shifted_status": shifted.status,
            "shifted_alerts": shifted.alerts,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    _write_json(work / "demo_summary.json", summary)
    print(json.dumps({"batch_scoring": summary["batch_scoring"], "drift": summary["drift"]}, indent=2))
    print(f"summary written to {work / 'demo_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdm", description="Remaining useful life lifecycle")
    p.add_argument("--tracking-uri", help="overrides PDM_TRACKING_URI, for example databricks")
    p.add_argument("--registry-uri", help="overrides PDM_REGISTRY_URI, for example databricks-uc")
    p.add_argument("--experiment", help="overrides PDM_EXPERIMENT")
    p.add_argument("--model-name", help="overrides PDM_MODEL_NAME, catalog.schema.model on Unity Catalog")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write a synthetic training fleet and a truncated holdout fleet")
    g.add_argument("--output-dir", required=True)
    g.add_argument("--units", type=int, default=100)
    g.add_argument("--holdout-units", type=int, default=300)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("features", help="build the training feature table")
    f.add_argument("--data-dir", required=True)
    f.set_defaults(func=cmd_features)

    t = sub.add_parser("train", help="train, log to MLflow and write a handoff file")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--handoff", required=True)
    t.add_argument("--model-type", choices=["hgb", "ridge"], default="hgb")
    t.add_argument("--max-iter", type=int, default=300)
    t.add_argument("--learning-rate", type=float, default=0.05)
    t.add_argument("--max-leaf-nodes", type=int, default=31)
    t.add_argument("--run-name", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate-register", help="register the challenger and apply the promotion gate")
    e.add_argument("--data-dir", required=True)
    e.add_argument("--handoff", required=True)
    e.add_argument("--max-rmse", type=float, default=GatePolicy.max_rmse)
    e.add_argument("--min-improvement", type=float, default=GatePolicy.min_relative_improvement)
    e.add_argument("--max-score-regression", type=float, default=GatePolicy.max_score_regression)
    e.set_defaults(func=cmd_evaluate_register)

    s = sub.add_parser("score", help="batch score telemetry with the champion")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--model-uri", default=None)
    s.add_argument("--threshold", type=float, default=30.0)
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("drift", help="data and prediction drift report")
    d.add_argument("--reference", required=True)
    d.add_argument("--current", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--thresholds", default=None)
    d.add_argument("--model-uri", default=None)
    d.add_argument("--no-predictions", action="store_true")
    d.add_argument("--fail-on-alert", action="store_true")
    d.set_defaults(func=cmd_drift)

    m = sub.add_parser("demo", help="run the full lifecycle locally")
    m.add_argument("--workdir", default=".local")
    m.add_argument("--units", type=int, default=100)
    m.add_argument("--holdout-units", type=int, default=300)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--threshold", type=float, default=30.0)
    m.add_argument("--shift", type=float, default=3.0)
    m.add_argument("--thresholds", default="monitoring/thresholds.yml")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run a subcommand. A non-zero result raises SystemExit, so a job task fails visibly."""
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    if code:
        raise SystemExit(code)
    return 0


if __name__ == "__main__":
    main()
