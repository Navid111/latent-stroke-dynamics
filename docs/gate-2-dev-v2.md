# Gate 2 development v2 — reviewed 2026-08-20

## Status

**The implementation is sound and one-step average-error prediction is strong, but exact action-grounded retrieval remains unresolved. This is development evidence, not a formal Gate 2 decision.**

The revised local suite passed (`15 passed in 1.99s`). The 256/64/96 development run completed on the base-model Apple Silicon MacBook Air without a runtime or memory failure. All rendered and encoded retrieval candidates were verified unique, all metrics were finite, and the implementation sanity check passed.

## Configuration

- frozen `facebook/dinov2-small`;
- 64×64 grayscale canvases and a 16×16 patch-token grid;
- 256 train, 64 validation, and 96 test transitions;
- permanently development-only data seeds `20260820`–`20260822`;
- model seed `11`;
- 30 maximum epochs, patience 6;
- linear predictor: 151,680 parameters;
- MLP predictor: 199,808 parameters;
- counterfactual cache version 2 with four distinct outcomes.

## Average-error prediction

Validation selected the linear predictor. Its validation and test errors were closely matched, with no evidence of a harmful generalization gap.

| Predictor | Test action-region MSE | Improvement over identity | Action-region next-token cosine distance |
|---|---:|---:|---:|
| Identity | 0.002270 | 0.0% | 0.4359 |
| Mean delta | 0.002003 | 11.8% | 0.3880 |
| Linear | 0.000973 | 57.2% | 0.1995 |
| MLP | 0.001095 | 51.8% | 0.2180 |

The selected linear model also improved by 51.4% over mean delta. Both learned models exceeded the frozen aggregate 30% error margin in this development run.

## Crowding behavior

| Prior strokes | Linear improvement over identity | MLP improvement over identity |
|---:|---:|---:|
| 0 | +77.3% | +76.0% |
| 5 | +35.2% | +22.5% |
| 15 | +9.3% | +1.8% |

The linear model was positive at every crowding level. The small high-crowding margin is partly explained by the identity baseline already being strong there: a new stroke produces a smaller latent change on a cluttered or partially occluded canvas.

## Convergence

The linear validation curve flattened near its best value around epoch 27. The MLP continued to improve slowly through epoch 30 but remained worse than linear on the validation action-region objective. The train/validation gaps were small, so simply adding many more epochs is not a justified retrieval repair.

## Counterfactual retrieval

The selected linear predictor retrieved the true outcome for 22 of 96 examples, or 22.9%. The MLP was similarly below the 25% random-choice reference; both trivial baselines were below 10%. The learned models therefore contain more action information than identity or mean delta, but they do not reliably identify the precise true outcome and are far below the frozen 50% requirement.

The shared-scale residual heatmap helps explain the split result:

- the predictor captures the broad spatial pattern of the latent change;
- its predicted residual is smoother and generally lower-amplitude than the true residual;
- the largest remaining error is concentrated near the proposed stroke;
- it can therefore reduce average MSE substantially while preferring a thinner, weaker, or slightly shifted counterfactual.

This is consistent with deterministic MSE regression toward an average consequence, not with an end-to-end implementation failure.

## Next diagnostic

Before freezing the formal command, decompose the existing retrieval CSV without retraining:

- candidate-selection rates;
- true-versus-shift, true-versus-width, and true-versus-intensity pairwise win rates;
- true-margin distribution and Wilson interval;
- retrieval by crowding, stroke width, intensity, and length.

`experiments/02b_retrieval_diagnostics.py` performs this analysis directly from the existing output directory. It cannot alter the model or produce a gate decision. Its purpose is to distinguish broad action insensitivity from a narrower residual-magnitude/calibration failure.
