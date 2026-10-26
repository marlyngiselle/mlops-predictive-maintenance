# ADR 0002. Champion and challenger with registry aliases and an automated gate

| Status | Date |
| --- | --- |
| Accepted | 2026-10-26 |

## Context

Unity Catalog does not have the old workspace registry stages (Staging, Production). It has aliases on model versions. Something has to decide which version serves, and that decision has to be reproducible, explainable and hard to bypass.

## Decision

- Every trained model is registered and receives the `challenger` alias.
- `pdm evaluate-register` scores the challenger and the current `champion` on the same holdout, in the same run, then applies `decide()` from `src/pdm/registry.py`.
- On success, the old champion becomes `previous_champion` and the challenger becomes `champion`. On failure, nothing that serves changes.
- The decision and every reason are written as tags on the model version.
- The release service principal owns the registered model in prod, so it is the only identity that can move aliases.

## Consequences

- Rollback is an alias swap (`pdm.registry.rollback`), seconds instead of a retrain.
- Batch scoring loads `models:/<name>@champion` at run time, so a promotion takes effect on the next run without redeploying anything.
- The champion is re-scored on today's holdout, not trusted on its old metrics. A champion that has aged badly makes it easier for a challenger to win, which is the desired behaviour.
- The gate is a pure function with unit tests for each rule, and an integration test that moves real aliases on a local SQLite registry.
- A human who disagrees with the gate changes the policy through a pull request. There is no manual promote button.
