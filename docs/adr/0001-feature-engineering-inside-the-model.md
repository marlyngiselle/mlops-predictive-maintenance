# ADR 0001. Feature engineering travels inside the model

| Status | Date |
| --- | --- |
| Accepted | 2026-10-26 |

## Context

Remaining useful life depends on history, meaning rolling statistics, trends and the deviation from a unit's own early life baseline. The classic failure mode is two implementations of these features, one in the training pipeline and one in the serving code, that slowly drift apart. The model then scores inputs it has never seen, and nothing fails loudly.

## Decision

The registered model is an MLflow pyfunc logged as code (`src/pdm/mlflow_model.py`). Its input is raw telemetry, its output is a prediction per row. Inside, it calls the same `build_features` function the training job uses, with the feature configuration saved next to the estimator. The `pdm` package ships in the model through `code_paths`.

## Consequences

- Batch scoring, the Azure ML endpoint and the Databricks serving alternative all send raw telemetry. None of them knows what a feature is.
- The signature in Unity Catalog describes the plant contract (unit, cycle, settings, 13 sensors), which is what integrators need to read.
- Callers must send enough history for the windows (30 cycles). The scoring script documents this and the model still answers with fewer rows, using shorter windows.
- Feature computation runs on every request. At this scale (tens of units per call) it is milliseconds. A feature store becomes worth it when features are shared across models or windows grow to days of high frequency data.
- Models from code avoid pickling a Python class, which keeps the artifact loadable across refactors of the package.
