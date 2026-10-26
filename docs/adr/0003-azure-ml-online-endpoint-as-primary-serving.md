# ADR 0003. Azure ML managed online endpoint as the primary online serving path

| Status | Date |
| --- | --- |
| Accepted | 2026-10-26 |

## Context

Training, the registry and batch scoring run on Databricks. Online scoring is needed for maintenance planners and plant systems that ask "how long does this unit have" during a shift. Two options were assessed, both documented in `serving/`.

| Criterion | Azure ML managed online endpoint | Databricks Model Serving |
| --- | --- | --- |
| Network | Workspace private endpoint, managed network with approved outbound only, all in `infra/terraform/azureml` | Private Link to the serving plane, depends on the Databricks network setup |
| Release control | Blue and green deployments with explicit traffic split and per deployment smoke tests | Traffic split between served entities |
| Model source | Champion copied from Unity Catalog at release time | Reads the Unity Catalog version directly |
| Observability | Application Insights, Azure Monitor, data collector for inputs and outputs | Inference tables in Unity Catalog |
| Consumers | Plant integration already runs on Azure private networking | Data and analytics teams on Databricks |

## Decision

The Azure ML managed online endpoint is the primary online path, because consumers sit on the private Azure network and the release team needs blue and green control with its own approval. Databricks Model Serving stays as a documented alternative in `serving/databricks/model-serving.yml`, ready to include in the bundle.

## Consequences

- The release step copies the champion artifact from Unity Catalog into the Azure ML registry and records the source version as a tag. Unity Catalog stays the single source of truth for what is champion.
- The scoring script loads the MLflow model as is. Because features live in the model (ADR 0001), no feature code is duplicated in `score.py`.
- Two registries hold the model. The tag `source_version` on the Azure ML model and the `gate_*` tags in Unity Catalog keep the link auditable.
- If consumers move to Databricks, switching is one bundle include and no model change.
