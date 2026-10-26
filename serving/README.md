# Serving

Two ways to serve the champion online. Azure ML is the primary path ([ADR 0003](../docs/adr/0003-azure-ml-online-endpoint-as-primary-serving.md)). Both load the same MLflow model, which takes raw telemetry and computes its own features.

| Path | Files | When |
| --- | --- | --- |
| Azure ML managed online endpoint (primary) | `azureml/endpoint.yml`, `azureml/deployment.yml`, `azureml/score.py`, `azureml/environment/conda.yml` | Consumers on the private Azure network, blue and green releases |
| Databricks Model Serving (alternative) | `databricks/model-serving.yml` | Consumers on Databricks, inference tables in Unity Catalog |

## Release the champion to Azure ML

The names below are the Terraform defaults for prod (`infra/terraform/azureml`).

```bash
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_REGISTRY_URI=databricks-uc

# 1. Copy the Unity Catalog champion locally, with its version for the audit trail
SOURCE_VERSION=$(python -c "from mlflow import MlflowClient; print(MlflowClient().get_model_version_by_alias('ml_prod.pdm.rul_model', 'champion').version)")
mlflow artifacts download --artifact-uri "models:/ml_prod.pdm.rul_model@champion" --dst-path ./champion

# 2. Register it in the Azure ML workspace
az ml model create --name pdm-rul --type mlflow_model --path ./champion \
  --tags source=unity-catalog source_version="$SOURCE_VERSION" \
  --resource-group rg-pdm-prod-ml --workspace-name mlw-pdm-prod

# 3. First time only: the endpoint
az ml online-endpoint create --file serving/azureml/endpoint.yml \
  --resource-group rg-pdm-prod-ml --workspace-name mlw-pdm-prod

# 4. New deployment at 0 percent traffic, smoke test it, then switch
az ml online-deployment create --file serving/azureml/deployment.yml \
  --set name=green model=azureml:pdm-rul@latest \
  --resource-group rg-pdm-prod-ml --workspace-name mlw-pdm-prod
az ml online-endpoint invoke --name pdm-rul --deployment-name green \
  --request-file serving/azureml/sample-request.json \
  --resource-group rg-pdm-prod-ml --workspace-name mlw-pdm-prod
az ml online-endpoint update --name pdm-rul --traffic "blue=0 green=100" \
  --resource-group rg-pdm-prod-ml --workspace-name mlw-pdm-prod
```

Rollback is the same traffic command in reverse, while `blue` still exists.

## Request and response

`sample-request.json` holds two cycles of a healthy unit, and the demo champion answers with the 125 cycle cap shown below. In practice, send the last 30 cycles of each unit so the rolling windows are full.

```json
{"predictions": [{"unit_id": 7, "cycle": 2, "predicted_rul": 125.0, "maintenance_due": false}]}
```

`maintenance_due` is true when the prediction is at or under `MAINTENANCE_THRESHOLD_CYCLES` (30 by default, set in `deployment.yml`).

## Tested locally

`tests/test_serving_and_scoring.py` downloads a registered champion into an `AZUREML_MODEL_DIR` layout, imports `score.py`, calls `init()` and `run()` and checks the answer. It also checks that the sample request matches the model signature column for column.
