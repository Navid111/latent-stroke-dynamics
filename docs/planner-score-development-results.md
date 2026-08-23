# Planner-score long-horizon development results

## Status

The single authorized three-target planner-development comparison completed and is now closed. Implementation integrity passed, all deterministic replays passed, every model stayed frozen, no closed target was reused, and no training or fine-tuning occurred.

The frozen decision is `not_eligible`. Confirmatory evaluation remains unauthorized.

## Aggregate result

| Method | Mean final MSE | Mean best MSE | Mean best step | Mean executed steps |
| --- | ---: | ---: | ---: | ---: |
| Exact pixel | 0.046674 | 0.046670 | 99.0 | 100.0 |
| Learned pixel | 0.058761 | 0.055860 | 67.3 | 100.0 |
| Current latent MSE, forced | 0.077652 | 0.072546 | 66.7 | 100.0 |
| Selected latent L1, forced | 0.072176 | 0.068289 | 71.0 | 100.0 |
| Selected latent L1, zero-margin no-op | 0.137607 | 0.137607 | 3.3 | 3.3 |

## What improved

Changing only the planner score from normalized-latent MSE to normalized-latent L1 reduced mean final MSE by about 7.05% under the same frozen MSE-only predictors and forced 100-step horizon. L1 had lower final MSE on all three development targets. It also improved mean best MSE by about 5.87%, mean exact rank from 23.11 to 21.66, and mean exact regret from 0.001407 to 0.001351.

This is useful exploratory evidence that planner-score alignment matters beyond the independent state audit. It does not establish a confirmatory result. The selected forced L1 planner remained weaker than learned pixel and ended at about 1.55 times exact-pixel error on these three development targets.

The per-step candidate diagnostics were mixed: forced L1 improved mean rank and regret but had lower top-1, top-5, and mean Spearman values than forced latent MSE. Long-horizon final performance therefore cannot be reduced to one local ordering metric alone.

## Why the no-op failed

The zero-margin no-op stopped after an average of only 3.33 strokes: six on target 1, zero on target 2, and four on target 3. Its mean final MSE was 0.137607, which was 77.2% higher than the current forced latent-MSE baseline. Target 2 stopped before drawing any stroke, so the method did not strictly improve every target from blank.

The stopping comparison was not calibrated: it compared the exact observed current-state latent distance with model-predicted candidate latent distances. Predictor error can shift all candidate scores upward, making the untouched current state appear preferable even when exactly rendered candidates would improve the image. The rule therefore confused score-scale mismatch with genuine convergence.

The no-op method's high top-5 rate applies only to the very small set of strokes executed before stopping and must not be interpreted as strong overall planning performance.

## Frozen decision

Two eligibility criteria passed:

- implementation integrity;
- selected predictor/score pair matched the archived audit.

Two eligibility criteria failed:

- selected no-op did not improve every target from blank;
- selected no-op did not reduce mean final MSE versus current forced latent MSE.

Therefore the preregistered confirmatory phase is not eligible and must not run.

## Thesis interpretation

The Stage A extension produced a nuanced result:

1. score alignment helped: normalized-latent L1 modestly but consistently improved the forced latent planner without retraining;
2. a naive no-op was harmful because exact current-state and predicted candidate scores were not calibrated;
3. stronger stopping requires an explicitly calibrated progress or value model, not a direct comparison between unlike score distributions;
4. learned pixel remained substantially stronger, while exact pixel remained the upper reference.

This supports reporting score alignment as an exploratory positive result and stopping calibration as a clear negative result. It does not justify changing the frozen decision or using the reserved confirmatory targets.

## Immutable boundaries

- Do not rerun or retune this planner-development comparison.
- Do not tune the no-op margin on these targets.
- Do not run the reserved confirmatory phase.
- Keep the local complete artifacts under `outputs/planner-score-audit-planner-development-2026-08-23/`.
- Preserve the `not_eligible` decision and both positive and negative findings.
