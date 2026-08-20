# Gate 2 protocol — deterministic one-step latent prediction

**Status:** Frozen before implementation and before any Gate 2 result  
**Frozen on:** 2026-08-20  
**Active research question:** Can a small action-conditioned model predict the spatial representation of the true next canvas better than trivial baselines on held-out one-stroke transitions?

## 1. Why this gate exists

Gate 1 established that frozen DINOv2-small patch features preserve a controlled line stroke in the correct spatial region, including under synthetic crowding. Gate 2 tests the next independent claim: whether the effect of a proposed stroke can be predicted from the current frozen representation and the action.

This is a one-step prediction experiment, not yet a painting planner. The renderer remains the source of truth.

## 2. Provenance and project decisions

The literature report motivates synthetic transition triples, frozen feature objectives, latent next-state prediction, and strict comparisons with trivial baselines. The following are project decisions rather than literature facts:

- start with deterministic dynamics because the renderer is deterministic and fully observed;
- predict DINOv2 spatial patch-token residuals because Gate 1 showed global pooling dilutes local changes;
- use a 30% improvement threshold as a practical go/no-go margin;
- require a counterfactual retrieval check so low average error cannot hide action-insensitive predictions.

Stochastic or mixture-density dynamics require a later ablation and are not part of this gate.

## 3. Frozen scope

- Canvas: 64×64 grayscale.
- Primitive: one deterministic straight line.
- Ground-truth transition: `C_next = R(C_current, stroke)`.
- Encoder: `facebook/dinov2-small`, frozen.
- State target: final-layer spatial patch tokens; the global token is diagnostic only.
- Prediction form: residual, `z_hat_next = z_current + delta_hat`.
- No reinforcement learning.
- No target-guided candidate selection.
- No multi-step rollout.
- No encoder fine-tuning.

## 4. Transition distribution

Each example contains:

```text
(current canvas, proposed stroke, true next canvas)
```

### Current canvas

The current canvas contains a randomly sampled number of prior strokes. Primary in-distribution crowding levels are:

```text
0, 5, 15 prior strokes
```

The level is sampled approximately uniformly within each split.

### Proposed stroke

Primary in-distribution actions use:

- normalized random endpoints within the canvas;
- minimum normalized length `0.20`;
- width sampled from `{1, 2, 3, 4}` pixels;
- grayscale value sampled from `{0, 32, 64, 96, 128}`.

Examples that change no pixels are rejected and resampled.

### Formal split sizes and seeds

| Split | Examples | Data seed | Purpose |
|---|---:|---:|---|
| Train | 1,000 | `20260820` | Fit trainable predictors and training-set mean delta |
| Validation | 200 | `20260821` | Model selection and early stopping only |
| Test | 300 | `20260822` | One final held-out evaluation |

The Gate 1 formal seed `20260819` is not reused. Splits are generated independently; no canvas, proposed action, or encoded target is shared across them.

### Secondary out-of-distribution stress slices

These do not decide the primary pass/fail result. Use 100 examples per slice with independent substreams from seed `20260823`:

1. unseen width: width `5`, otherwise primary settings;
2. unseen intensity: values from `{16, 80, 176}`, otherwise primary settings;
3. unseen crowding: `10` prior strokes, otherwise primary settings.

## 5. Frozen representation and action encoding

For each canvas, extract normalized DINOv2 patch tokens:

```text
z_current = E(C_current)
z_next    = E(C_next)
delta     = z_next - z_current
```

The predictor receives, per spatial patch:

1. the current patch token;
2. a normalized global stroke vector containing midpoint, undirected orientation, length, width, and darkness;
3. fractional patch coverage from a rasterized action mask;
4. normalized patch coordinates.

The action mask is an input, not a target-derived change mask. It may be computed before the stroke is committed.

## 6. Predictors and baselines

All baselines remain visible in the final table.

1. **Identity / no-change:** `delta_hat = 0`.
2. **Mean delta:** the training-set mean residual, never recomputed on validation or test.
3. **Linear action-conditioned predictor:** one shared affine mapping applied patch-wise.
4. **Small nonlinear deterministic predictor:** a shared patch-wise MLP with at most about one million trainable parameters.

The linear model is a legitimate positive result if it wins. Gate 2 tests predictability, not whether nonlinearity is necessary.

Use three model-initialization seeds for each trainable predictor:

```text
11, 22, 33
```

## 7. Training objective

Use a spatially balanced residual mean-squared error:

```text
L = 0.5 * MSE_inside_action_region
  + 0.5 * MSE_outside_action_region
```

The balancing prevents the many unchanged patches from overwhelming the few action-aligned patches. Hyperparameters and early stopping are selected on validation data only.

A tiny overfit check on a deliberately small training subset must be run before the ordinary smoke test. It is an implementation check, not evidence for the gate.

## 8. Primary and secondary metrics

### Primary

- action-region residual MSE;
- relative improvement over identity and mean-delta baselines;
- counterfactual retrieval accuracy: whether the predicted next representation is closest to the true post-stroke result rather than shifted, width-changed, or intensity-changed alternatives.

### Secondary

- all-patch residual MSE;
- outside-action-region residual MSE;
- cosine distance between predicted and true next patch tokens;
- error by crowding, width, intensity, and stroke length;
- linear versus nonlinear comparison;
- out-of-distribution stress-slice performance;
- training curves and qualitative residual-error heatmaps.

The objective encoder is not used to claim final painting quality. Independent image-space evaluation and exact-renderer comparisons enter Gate 3.

## 9. Counterfactual retrieval diagnostic

For each held-out test transition, retain the same current canvas and construct four exact rendered outcomes:

1. the true proposed stroke;
2. a position-shifted stroke;
3. a width-changed stroke;
4. an intensity-changed stroke.

Encode all four exact outcomes. Score each candidate over the union of their action-covered patches and ask whether `z_hat_next` retrieves the true outcome at rank 1. Chance is 25%.

This is a grounding diagnostic, not target-guided planning. The model is not choosing a stroke toward a target image yet.

## 10. Frozen decision rule

Evaluate the test set once after code, smoke checks, hyperparameters, and model choice are frozen.

### Pass

Gate 2 passes if the best trainable predictor, averaged over its three initialization seeds:

1. reduces primary action-region MSE by at least **30%** relative to both identity and mean-delta baselines;
2. has positive improvement over identity at every primary crowding level (`0`, `5`, and `15`);
3. achieves at least **50%** counterfactual top-1 retrieval accuracy, twice the 25% chance rate;
4. shows no implementation sanity failure or unstable seed collapse.

The predictor does not have to beat the linear model by 30%. If the linear model satisfies the rule, the conclusion is that one-step latent dynamics are predictably simple.

### Borderline

A result is borderline if improvement is 10–30%, retrieval is 35–50%, or one crowding level fails while aggregate performance passes. Report it as borderline and perform only diagnostics already named in this protocol; do not tune on the test set.

### Fail

Gate 2 fails if the best trainable predictor improves by less than 10% over a trivial baseline, retrieval remains at or below 35%, results collapse across seeds, or predictions ignore the proposed action.

## 11. Pixel-space control

A small action-conditioned pixel-space next-canvas predictor is required before the final thesis comparison because prior work establishes pixel prediction as the known-working alternative. It is not part of the initial latent-pipeline smoke test and does not change this gate's primary question. Add it after the latent implementation is validated and before beginning Gate 3.

## 12. Required artifacts

A reproducible Gate 2 run must save:

- `run_config.json`;
- split metadata and fingerprints;
- cached encoded features or clear cache metadata;
- per-example prediction metrics;
- aggregate metrics by model, seed, split, and crowding;
- training history;
- counterfactual retrieval results;
- gate diagnostics;
- at least one spatial residual/error visualization.

Generated datasets, feature caches, model checkpoints, and raw outputs remain ignored by Git. Curated final tables and figures may be copied into `results/` after the formal decision.

## 13. Execution order

1. Commit this protocol.
2. Implement deterministic data generation, action encoding, predictors, metrics, and unit tests.
3. Run `pytest`.
4. Run a tiny overfit check.
5. Run a small smoke experiment and repair implementation only.
6. Freeze the formal command/config in a separate commit.
7. Run the held-out formal experiment.
8. Begin Gate 3 only after a recorded Gate 2 pass.
