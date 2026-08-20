# Current State

**Last updated:** 2026-08-20  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Implementation committed; local validation next

## Objective

Train the smallest action-conditioned model that predicts how one deterministic stroke changes frozen spatial canvas features, and determine whether it outperforms trivial baselines on held-out transitions.

## Completed

- Implemented deterministic rendering, frozen DINOv2 encoding, controlled Gate 1 interventions, plots, metrics, and tests.
- Froze and ran the formal Gate 1 experiment with seed `20260819`.
- Archived the formal artifacts and declared Gate 1 a pass against its frozen criteria.
- Read the focused literature-report scope and reconciled it with the project design.
- Froze `docs/gate-2-protocol.md` before implementing Gate 2.
- Added deterministic Gate 2 transition generation and split fingerprints.
- Added a reversal-invariant normalized stroke vector, fractional patch-action mask, and normalized patch coordinates.
- Added identity, mean-delta, shared linear, and small shared nonlinear residual predictors.
- Added balanced action/outside loss, spatial errors, and true-versus-counterfactual retrieval.
- Added Gate 2 unit tests for generation, encodings, masks, tensor shapes, loss, and retrieval.
- Added an M1-aware experiment pipeline with chunked encoding, float16 disk caches, float32 training batches, tiny overfit check, early stopping, output tables, and plots.

## Formal Gate 1 result

| Crowding | Reference-region wins | Median localization lift | Median reference enrichment | No-change max |
|---:|---:|---:|---:|---:|
| 0 | 25/25 (100%) | 12.80× | 2.05× | 2.98e-7 |
| 5 | 24/25 (96%) | 10.24× | 4.95× | 3.58e-7 |
| 15 | 25/25 (100%) | 10.60× | 6.77× | 3.58e-7 |

The spatial representation preserves the stroke in the correct action region. Global and fixed top-10% summaries weaken under clutter, so Gate 2 uses patch tokens and action-aligned metrics.

See `docs/gate-1-results.md` for the complete interpretation.

## Frozen Gate 2 decisions

- Keep `facebook/dinov2-small` frozen.
- Predict spatial patch-token residuals: `delta = z_next - z_current`.
- Use deterministic one-step transitions.
- Encode the stroke with normalized parameters, a patch-aligned action mask, and patch coordinates.
- Compare identity, mean-delta, linear, and small nonlinear predictors.
- Use independent train, validation, test, and stress seeds.
- Train each learned predictor with seeds `11`, `22`, and `33` for the formal result.
- Use action-region error and counterfactual retrieval as primary evidence.
- Do not begin target-guided ranking, reinforcement learning, stochastic dynamics, or multi-step rollout.

## Validation status

No claim has been made that the new Gate 2 tests or experiment pass locally. The repository-side implementation is ready for Navid's machine, where the actual Python environment and M1 resource behavior must be checked.

A smoke-sized run is intentionally ineligible for a gate decision and reports `diagnostic_only`. The formal command remains unfrozen until smoke validation is complete.

## Next actions

1. Pull the latest `main` on the local M1.
2. Run the full `pytest` suite and preserve the complete result.
3. Run the tiny-overfit plus smoke experiment from `README.md`.
4. Inspect `overfit_check.json`, `gate_diagnostics.csv`, the training history, and both plots.
5. Repair implementation errors if present without changing the frozen scientific criteria.
6. Freeze the exact formal Gate 2 command only after the smoke check is sound.

## Immediate next step

Navid should run `git pull`, then `pytest`, then the committed Gate 2 smoke command. Gate 1 must not be rerun or retuned.

## Handoff note

The literature report supports the direction but does not override empirical gate results. Deterministic-first dynamics, patch-token targets, exact-renderer grounding, and the numerical Gate 2 decision rule are explicit project decisions. Keep provenance clear in the thesis.
