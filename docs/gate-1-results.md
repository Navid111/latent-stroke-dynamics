# Gate 1 formal result

**Date:** 2026-08-19  
**Decision:** **PASS**  
**Result commit:** `ab499638d03efeff58e771392cbb135851346286`  
**Artifacts:** `results/gate1-formal/2026-08-19/`

## Question

Does a frozen spatial visual representation preserve enough information about one controlled stroke to justify training an action-conditioned next-representation predictor?

## Formal configuration

- encoder: `facebook/dinov2-small`
- representation: 16×16 spatial patch grid
- canvas: 64×64 grayscale
- canonical action: black straight line, width 2, minimum normalized length 0.35
- samples per crowding level: 25
- nested crowding: 0, 5, and 15 prior strokes
- conditions: 8
- total comparison pairs: 600
- seed: `20260819`
- device: CPU
- primary nuisance control: sparse random perturbation with exact changed-pixel-count and pixel-MAE matching

## Primary findings

| Crowding | Reference-region win rate | Median localization top-k lift | Median reference enrichment | No-change maximum |
|---:|---:|---:|---:|---:|
| 0 | 1.00 (25/25) | 12.80× | 2.05× | 2.98e-7 |
| 5 | 0.96 (24/25) | 10.24× | 4.95× | 3.58e-7 |
| 15 | 1.00 (25/25) | 10.60× | 6.77× | 3.58e-7 |

Every frozen primary threshold was met. Moderate crowding did not erase the coherent stroke from its action region, and the unrequired crowding-15 stress test was also strongly positive.

## Magnitude versus localization

The absolute representation change became smaller as clutter increased:

| Crowding | Added-stroke mean reference-region distance | Sparse-control mean reference-region distance | Added-stroke global distance |
|---:|---:|---:|---:|
| 0 | 0.746 | 0.448 | 0.359 |
| 5 | 0.310 | 0.122 | 0.049 |
| 15 | 0.218 | 0.068 | 0.024 |

This is an important limitation: global pooled features become much less sensitive to one new stroke on a busy canvas. Spatial patch features nevertheless retain a strong local signal. Gate 2 should therefore predict patch-level changes rather than relying only on a global token.

## Why fixed top-10% separation is low

The added stroke beat sparse matched noise on fixed top-10% patch distance in:

- 68% of blank samples,
- 8% of five-stroke samples,
- 4% of fifteen-stroke samples.

The sparse control changes the same number of pixels and total pixel intensity, but it scatters those pixels across roughly 20% of patch locations. The coherent line occupies roughly 5%. A fixed top-10% average therefore rewards dispersed high-frequency perturbations because they activate more patch cells.

This behavior was identified in the pilot and formally designated a secondary robustness metric before this run. It is not hidden: it shows that DINOv2 is sensitive to nuisance texture and that a planner should use spatially action-aligned losses rather than an unstructured top-k magnitude alone.

## Frozen-criterion decision

- numerical-zero sanity: **pass**
- paired reference-region separation at crowding 0: **pass**
- paired reference-region separation at crowding 5: **pass**
- localization lift at crowding 0: **pass**
- localization lift at crowding 5: **pass**
- reference-region enrichment at crowding 0 and 5: **pass**
- spatial heatmap/changed-region consistency: **pass**
- crowding-15 stress test: **strong positive result**

**Overall Gate 1 decision: PASS.**

## What this result supports

The formal evidence supports the following scoped claim:

> Frozen DINOv2 patch features reliably encode the local consequence of adding one simple stroke to a synthetic canvas, even when existing strokes create moderate or high clutter.

It does not yet support claims that:

- the next latent state is predictable from the current state and action,
- a learned predictor can outperform simple baselines,
- latent loss ranks candidate strokes correctly,
- multi-step planning works,
- DINOv2 is equivalent to JEPA,
- the finding generalizes to realistic brushes, color, texture, or high-resolution art.

## Gate 2 implications

1. Keep the encoder frozen.
2. Use spatial patch tokens as the prediction target.
3. Predict a residual, `Δz = z_next - z_current`, rather than reconstructing the full representation from scratch.
4. Include normalized action parameters and a patch-aligned action mask.
5. Compare no-change, mean-delta, linear, and small nonlinear predictors.
6. Use new, explicitly separated train/validation/test seeds.
7. Do not begin candidate ranking, reinforcement learning, or multi-step planning until one-step results justify it.
