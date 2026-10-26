# Monitoring and retraining policy

## What is watched

| Signal | Method | Warn | Alert | Source |
| --- | --- | --- | --- | --- |
| Each of the 13 sensors | PSI on 10 reference quantile bins | 0.10 | 0.25 | `monitoring/thresholds.yml` |
| Each of the 13 sensors | Two sample KS | | p below 0.01 and statistic at least 0.20 | `monitoring/thresholds.yml` |
| Predicted remaining life | PSI, same bins | 0.10 | 0.25 | `monitoring/thresholds.yml` |
| Job health | Databricks job failure | | Any failed task | Email from the job definition |
| Endpoint health | Azure ML endpoint metrics and Application Insights | | Latency or error rate | `infra/terraform/azureml` diagnostics |

The reference window is the promotion holdout, the population the champion was last validated on. The current window is the latest in-service snapshot.

KS needs both conditions on purpose. With tens of thousands of rows, a p-value is tiny for differences nobody cares about. In the measured run below, the stable window has a prediction KS p-value of 7e-12 with a statistic of 0.02, and it is correctly reported as ok.

## What the demo measured

`pdm demo` compares the holdout with a fresh in-service fleet drawn from the same process, then with the same fleet after a calibration fault of 3.0 units on `sensor_04` (about three times its noise).

| Scenario | Data drift | Largest PSI | Prediction PSI | Status |
| --- | --- | --- | --- | --- |
| Stable fleet | none | 0.009 | 0.008 | ok |
| Calibration fault on `sensor_04` | `sensor_04` only, PSI 2.835, KS 0.706 | 2.835 | 0.005 | alert |

The second row is the interesting one. The data alert fires on exactly one sensor, and the predictions do not move. The model leans on other sensors and on trends rather than on the absolute level of `sensor_04`. That combination says "fix the sensor", not "retrain the model", and the playbook below treats it that way. The full report is in [`monitoring/sample-alert-report.json`](../monitoring/sample-alert-report.json).

## When to retrain

| Trigger | Action | Who decides |
| --- | --- | --- |
| Weekly schedule | `pdm_training` runs. The gate decides whether anything changes | The gate |
| Data alert, predictions stable | Open an instrumentation ticket. No retrain, the model is not affected yet | On-call engineer |
| Data alert and prediction alert | Check the input first. If the shift is real (new operating regime, new fleet), trigger `pdm_training` manually | On-call engineer |
| Prediction alert alone | Investigate the population mix (more young or old units). Usually expected after fleet changes | On-call engineer |
| Ground truth arrives (a failure or a teardown) | Append to the training snapshot, next scheduled run picks it up | Pipeline |
| Champion degrades on new ground truth | Manual `pdm_training` run, then `rollback` if the new champion is worse | On-call engineer |

Retraining never bypasses the gate. A manual run goes through the same three tasks and the same policy as the weekly one.

## Changing a threshold

Thresholds live in `monitoring/thresholds.yml`, are copied to the volume by the release step, and are read by the drift job. A change is a pull request like any other, and `Thresholds.from_yaml` rejects unknown keys so a typo fails loudly instead of silently using a default.
