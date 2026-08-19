# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Gate 1 formally passed; Gate 2 design is next

## Objective

Train the smallest action-conditioned model that predicts how one deterministic stroke changes frozen spatial canvas features, and determine whether it outperforms trivial baselines on held-out transitions.

## Completed

- Implemented deterministic rendering, frozen DINOv2 encoding, paired controls, plots, metrics, and tests.
- Fixed environment and plotting compatibility issues.
- Ran three Gate 1 pilot iterations and used them only to repair implementation and finalize controls.
- Froze the formal Gate 1 protocol before the final run.
- Ran the formal Gate 1 experiment with seed `20260819`, 25 samples at each of three crowding levels, eight conditions, and 600 comparison pairs.
- Archived all formal artifacts under `results/gate1-formal/2026-08-19/`.
- Evaluated the formal run against the frozen criteria.
- Declared Gate 1 a pass.

## Formal Gate 1 result

| Crowding | Reference-region wins | Median localization lift | Median reference enrichment | No-change max |
|---:|---:|---:|---:|---:|
| 0 | 25/25 (100%) | 12.80× | 2.05× | 2.98e-7 |
| 5 | 24/25 (96%) | 10.24× | 4.95× | 3.58e-7 |
| 15 | 25/25 (100%) | 10.60× | 6.77× | 3.58e-7 |

The frozen spatial representation reliably preserves the stroke in the correct action region. Fixed top-10% and global-token scores weaken under clutter because sparse random pixels cover many more patch cells and pooled features dilute one local change. These limitations are documented rather than hidden.

See `docs/gate-1-results.md` for the complete interpretation.

## Gate 2 decisions already supported

- Keep `facebook/dinov2-small` frozen for the initial predictor experiment.
- Predict spatial patch-token residuals rather than relying only on the global token.
- Use deterministic one-step transitions.
- Encode the stroke with normalized action parameters and a patch-aligned action mask.
- Compare no-change, mean-delta, linear, and small nonlinear predictors.
- Use separate train, validation, and test seeds that are not the Gate 1 formal seed.
- Evaluate held-out prediction before candidate ranking.
- Do not begin reinforcement learning or multi-step planning.

## Next actions

1. Write and freeze a concise Gate 2 protocol: transition distribution, splits, inputs, targets, baselines, metrics, and pass rule.
2. Add a reproducible transition-dataset generator with explicit train/validation/test seeds.
3. Add baseline and residual-predictor implementations with unit tests.
4. Run a tiny overfit/smoke test to verify the learning pipeline.
5. Run the frozen held-out Gate 2 comparison.
6. Proceed to candidate-stroke ranking only if the learned predictor beats the frozen baselines.

## Immediate next step

No new local command is required yet. The next repository change should define and implement the Gate 2 experiment before starting another run.

## Handoff note

Gate 1 is complete and must not be rerun or retuned. The next agent should preserve its result, design Gate 2 with independent splits, and begin with a deterministic residual predictor. Candidate ranking and planning remain out of scope until one-step prediction passes.
