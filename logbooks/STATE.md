# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Formal configuration frozen; untouched formal run next

## Objective

Determine whether a small action-conditioned predictor can estimate one-stroke changes in frozen spatial canvas features well enough for later candidate-stroke planning.

## Completed

- Froze, ran, and passed Gate 1.
- Froze the Gate 2 scientific protocol before predictor implementation.
- Implemented and locally validated deterministic transitions, action encoding, four baselines/predictors, balanced loss, caching, metrics, retrieval, plots, and tests.
- Repaired duplicate counterfactual outcomes and transparently retired exposed development seed prefixes before formal evaluation.
- Completed development v2 on 256/64/96 examples.
- Observed 57.2% action-region improvement over identity, 51.4% over mean delta, and positive improvement at every crowding level.
- Observed 22/96 top-1 retrieval with unique candidates.
- Passed the 18-test suite in 3.55 seconds.
- Decomposed retrieval without retraining: position and intensity pairwise wins were 66.7%, while width pairwise wins were 46.9%.
- Determined that no remaining implementation defect justifies changing the primary experiment.
- Froze the exact formal command and configuration before generating amended formal data.

## Development interpretation

The predictor learns broad action-conditioned latent consequences and is not completely action-blind. Its main weakness is precise stroke-width discrimination, consistent with a smooth or lower-amplitude MSE prediction. Development evidence cannot decide Gate 2.

See:

- `docs/gate-2-dev-v2.md`;
- `docs/gate-2-retrieval-diagnostics.md`;
- `docs/gate-2-formal-config.md`.

## Frozen formal configuration

- Train/validation/test: 1,000/200/300.
- Stress: 100 examples per frozen slice.
- Data seeds: `20260824`, `20260825`, `20260826`, stress base `20260827`.
- Model seeds: `11`, `22`, `33`.
- Epochs: 30; patience: 6.
- Learning rate: 0.001; weight decay: 0.0001.
- Hidden size: 256; train batch: 16.
- CPU encoding and training on the M1.
- Exact command: `docs/gate-2-formal-config.md`.

## Next actions

1. Pull the formal-freeze commit.
2. Run the exact command once without modifying settings.
3. Verify `formal_eligible: true`.
4. Preserve all terminal output and generated artifacts.
5. Judge the result against the frozen rule without tuning.
6. Add the pixel-space control before Gate 3 or the final thesis comparison.

## Immediate next step

Run the untouched formal Gate 2 experiment. Do not change code, data seeds, hyperparameters, thresholds, or candidate construction first.

## Handoff note

A pass is not assumed. A mixed or failed result—strong average prediction but weak exact retrieval—is a valid thesis outcome and must be reported honestly.
