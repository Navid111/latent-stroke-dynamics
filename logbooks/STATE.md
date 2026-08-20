# Current State

**Last updated:** 2026-08-20  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Smoke 1 reviewed; integrity repairs committed; development v2 next

## Objective

Train the smallest action-conditioned model that predicts how one deterministic stroke changes frozen spatial canvas features, and determine whether it outperforms trivial baselines on held-out transitions.

## Completed

- Implemented deterministic rendering, frozen DINOv2 encoding, controlled Gate 1 interventions, plots, metrics, and tests.
- Froze and ran the formal Gate 1 experiment with seed `20260819`.
- Archived the formal artifacts and declared Gate 1 a pass against its frozen criteria.
- Read the focused literature-report scope and reconciled it with the project design.
- Froze `docs/gate-2-protocol.md` before implementing Gate 2.
- Added deterministic Gate 2 transitions, split fingerprints, normalized action inputs, four baselines/predictors, balanced loss, spatial metrics, retrieval, tests, and an M1-aware experiment pipeline.
- Ran the complete local suite: 14 tests passed in 1.91 seconds under Python 3.14.4 and pytest 9.1.1.
- Completed the first end-to-end Gate 2 engineering smoke on the local M1 without a runtime or memory failure.
- Confirmed a 98.14% tiny-overfit loss reduction and finite decreasing training curves.
- Observed promising aggregate linear-predictor error but mixed crowding behavior.
- Found and repaired duplicate counterfactual outcomes that invalidated the first smoke's nominal retrieval chance rate.
- Amended formal data seeds before any formal run because the original seed prefixes were exposed by the smoke command.
- Added stricter formal-eligibility, finite-metric, overfit, and candidate-uniqueness checks.
- Added crowding, retrieval, and training plots plus common-scale residual heatmaps.

## Formal Gate 1 result

| Crowding | Reference-region wins | Median localization lift | Median reference enrichment | No-change max |
|---:|---:|---:|---:|---:|
| 0 | 25/25 (100%) | 12.80× | 2.05× | 2.98e-7 |
| 5 | 24/25 (96%) | 10.24× | 4.95× | 3.58e-7 |
| 15 | 25/25 (100%) | 10.60× | 6.77× | 3.58e-7 |

See `docs/gate-1-results.md` for the complete interpretation.

## Gate 2 smoke 1

The 64/16/32, one-seed run was diagnostic only. Validation selected the linear predictor.

- action-region MSE improvement versus identity: 48.2%;
- improvement versus mean delta: 36.8%;
- crowding improvements versus identity: +67.7%, +6.3%, and −25.7% for 0, 5, and 15 prior strokes;
- nominal retrieval: 31.25%, but not interpretable because five rows contained exact candidate ties;
- overfit check: 98.14% loss reduction.

See `docs/gate-2-smoke-1.md` for the review and integrity corrections.

## Frozen Gate 2 decisions

- Keep `facebook/dinov2-small` frozen.
- Predict spatial patch-token residuals: `delta = z_next - z_current`.
- Use deterministic one-step transitions.
- Encode the stroke with normalized parameters, a patch-aligned action mask, and patch coordinates.
- Compare identity, mean-delta, linear, and small nonlinear predictors.
- Use independent train, validation, test, and stress seeds.
- Use formal model seeds `11`, `22`, and `33`.
- Use untouched amended formal data seeds `20260824`–`20260827`.
- Require four distinct rendered and encoded outcomes for counterfactual retrieval.
- Use action-region error and counterfactual retrieval as primary evidence.
- Do not begin target-guided ranking, reinforcement learning, stochastic dynamics, or multi-step rollout.

## Validation status

Smoke 1 validates the M1 execution path, encoder caching, overfit path, training loop, evaluation, and artifact generation. It does not validate the original retrieval number because candidate aliases were discovered during review.

The repair is committed but has not yet passed Navid's local tests. The formal command remains unfrozen, and the untouched amended formal data have not been generated or viewed.

## Next actions

1. Pull the integrity-repair commits.
2. Run the complete `pytest` suite; the new counterfactual-uniqueness test must pass.
3. Run the 256/64/96 development-v2 command from the README.
4. Inspect convergence, unique-candidate diagnostics, crowding behavior, and retrieval.
5. Freeze the exact formal training command only if development v2 is sound.
6. Run the formal configuration once against untouched data.

## Immediate next step

Navid should run `git pull`, `pytest`, and then the committed Gate 2 development-v2 command. Gate 1 must not be rerun or retuned.

## Handoff note

The literature report supports the direction but does not override empirical gate results. Deterministic-first dynamics, patch-token targets, exact-renderer grounding, and the numerical Gate 2 decision rule are explicit project decisions. Keep provenance clear in the thesis.
