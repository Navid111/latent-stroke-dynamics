# Gate 2 engineering smoke 1 — reviewed 2026-08-20

## Status

**Engineering smoke succeeded; scientific result is not eligible for a Gate 2 decision.**

The complete local suite passed before this run (`14 passed in 1.91s`). The end-to-end experiment then completed on the base-model Apple Silicon MacBook Air using CPU encoding and training, with no crash or memory-pressure report.

## Configuration

- DINOv2-small, frozen
- 64×64 grayscale canvases
- 64 training, 16 validation, and 32 test transitions
- development data seeds `20260820`, `20260821`, and `20260822`
- model seed `11`
- 8 epochs, patience 3
- linear and 256-hidden-unit MLP predictors
- counterfactual retrieval enabled

This configuration was deliberately too small to declare a pass or fail.

## Implementation checks

The four-example overfit diagnostic reduced balanced MSE from `0.0374769` to `0.0006982`, a **98.14% reduction**. Both ordinary training curves remained finite and decreased through epoch 8. The script generated all requested tables and figures.

## Diagnostic prediction results

Validation selected the linear predictor.

| Predictor | Test action-region MSE | Improvement over identity |
|---|---:|---:|
| Identity | 0.002469 | 0.0% |
| Mean delta | 0.002024 | 18.0% |
| Linear | 0.001278 | 48.2% |
| MLP | 0.001504 | 39.1% |

The selected linear predictor improved by **36.8% over mean delta** as well as 48.2% over identity. Its action-region next-token cosine distance was also lower than both trivial baselines.

### Crowding diagnostic

| Prior strokes | Linear improvement over identity |
|---:|---:|
| 0 | +67.7% |
| 5 | +6.3% |
| 15 | −25.7% |

The aggregate result is promising, but the 32-example test split is too small for a stable crowding conclusion. The negative high-crowding slice must not be hidden or interpreted away.

## Counterfactual issue found during review

Nominal counterfactual top-1 retrieval was 31.25%. Inspection found exact true-versus-width score ties across every model on five of the 32 rows, indicating candidate-outcome aliases. Four distinct candidates are required for the stated 25% chance level, so this retrieval number is not scientifically interpretable.

The implementation now:

1. requires every counterfactual to change the current canvas;
2. requires all four rendered outcomes to be pixel-distinct;
3. deterministically tries another action from the same semantic class after rasterization or occlusion aliasing;
4. verifies that the four cached encoded outcomes are also distinct;
5. versions the counterfactual cache so the invalid cache cannot be silently reused.

This is an implementation-integrity repair, not threshold tuning.

## Formal-seed hygiene amendment

The first smoke command inadvertently used prefixes from the originally reserved formal data seeds. Because those rows are now visible, seeds `20260820`–`20260822` are permanently development-only. Before any formal run, the protocol was transparently amended to reserve untouched seeds:

- train: `20260824`
- validation: `20260825`
- test: `20260826`
- stress base: `20260827`

No pass threshold, model family, loss, metric, or model-initialization seed changed.

## Visualization review

The residual heatmap showed that one local stroke changes contextualized DINO patch tokens beyond the literal stroke mask. The predictor captured the coarse spatial pattern, while its largest errors remained near the action region. The first version used separate color scales for true and predicted residuals; the revised script uses one shared scale and adds crowding, retrieval, and training-curve plots.

## Next development check

Run the larger 256/64/96 development configuration with 30 maximum epochs and one model seed. It continues to use the exposed development seeds and remains ineligible for a formal decision. Its purpose is to validate the repaired retrieval set and determine convergence using validation curves—not to tune on a formal test set.
