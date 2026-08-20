# Current State

**Last updated:** 2026-08-20  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Protocol frozen; implementation is next

## Objective

Train the smallest action-conditioned model that predicts how one deterministic stroke changes frozen spatial canvas features, and determine whether it outperforms trivial baselines on held-out transitions.

## Completed

- Implemented deterministic rendering, frozen DINOv2 encoding, controlled Gate 1 interventions, plots, metrics, and tests.
- Ran pilot iterations only to repair implementation and finalize Gate 1 controls.
- Froze and ran the formal Gate 1 experiment with seed `20260819`.
- Archived all formal artifacts under `results/gate1-formal/2026-08-19/`.
- Declared Gate 1 a pass against the frozen criteria.
- Read the focused literature-report scope: all per-paper JEPA-relevance and takeaway sections plus Part III through the end.
- Reconciled the report with the formal Gate 1 evidence.
- Froze `docs/gate-2-protocol.md` before implementation or Gate 2 results.

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
- Train each learned predictor with three initialization seeds for the formal result.
- Use action-region error and counterfactual retrieval as primary evidence.
- Do not begin target-guided ranking, reinforcement learning, stochastic dynamics, or multi-step rollout.

## Next actions

1. Implement the Gate 2 transition generator and split metadata.
2. Add action encoding, feature-cache support, baseline predictors, and balanced loss.
3. Add unit tests for deterministic generation, split separation, shapes, baselines, and metrics.
4. Add the Gate 2 experiment script and reproducible outputs.
5. Run a tiny overfit check and smoke experiment on the local M1.
6. Freeze the formal command only after implementation checks pass.

## Immediate next step

Create the Gate 2 implementation. Navid should not run a new experiment until that commit is ready. Gate 1 is complete and must not be rerun or retuned.

## Handoff note

The literature report supports the direction but does not override empirical gate results. Deterministic-first dynamics, patch-token targets, exact-renderer grounding, and the numerical Gate 2 decision rule are explicit project decisions. Keep provenance clear in the thesis.
