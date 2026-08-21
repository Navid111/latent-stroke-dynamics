# Gate 2 retrieval decomposition — 2026-08-21

## Status

This is a post-hoc decomposition of the fixed development-v2 outputs. It did not load the encoder, retrain a model, alter a cache, or inspect formal data.

The accompanying diagnostic implementation passed locally as part of the full suite:

```text
18 passed in 3.55s
```

## Top-1 retrieval

| Model | Correct | Accuracy | 95% Wilson interval |
|---|---:|---:|---:|
| Identity | 8/96 | 8.3% | 4.3–15.6% |
| Mean delta | 7/96 | 7.3% | 3.6–14.3% |
| Linear | 22/96 | 22.9% | 15.6–32.3% |
| MLP | 20/96 | 20.8% | 13.9–30.0% |

The learned predictors retrieve the true consequence much more often than either trivial baseline, but remain statistically compatible with the 25% random-choice reference and far below the frozen 50% gate threshold.

## Candidate preferences

The linear predictor selected:

- true outcome: 22.9%;
- shifted position: 24.0%;
- changed width: 37.5%;
- changed intensity: 15.6%.

The MLP showed the same pattern and selected changed width 40.6% of the time. Width alternatives are therefore the dominant confusion, not intensity alternatives.

## Pairwise decomposition

| Model | True beats shifted | True beats width-changed | True beats intensity-changed |
|---|---:|---:|---:|
| Linear | 66.7% | 46.9% | 66.7% |
| MLP | 59.4% | 40.6% | 66.7% |

This rules out complete action blindness. The linear predictor distinguishes the true outcome from position and intensity counterfactuals on roughly two thirds of examples. Its primary bottleneck is stroke width, where performance is slightly below an even pairwise split.

The linear true-outcome margin has mean `-8.83e-05` and median `-8.28e-05`. Many errors are near misses, although a small tail of larger negative margins remains.

## Interpretation

Combined with the average-error and heatmap evidence, the most defensible interpretation is:

1. the model learns where a stroke changes the latent canvas;
2. it learns useful position and intensity information;
3. deterministic MSE produces a smoother, lower-amplitude residual;
4. that residual often resembles the consequence of a thinner or otherwise width-changed stroke more than the exact consequence.

Width is already present both as a normalized scalar and through fractional action-mask coverage. Unique candidate checks pass. No remaining implementation defect is evident that would justify changing the frozen task before formal evaluation.

## Decision before formal data

Do not add a width-specific loss, contrastive retrieval loss, new architecture, or revised threshold now. Those would be result-driven changes to the primary Gate 2 experiment. The correct next action is to freeze the existing training configuration and evaluate it once on the untouched formal splits with four times as many training examples and all three preregistered model seeds.

If formal retrieval remains weak, report a mixed or failed Gate 2 result: low average latent error is achievable, but the current deterministic patch-wise predictors are insufficiently precise for action-level planning. That is a valid bachelor's thesis finding and motivates the already-required pixel-space control or a clearly labeled future contrastive/spatial model.
