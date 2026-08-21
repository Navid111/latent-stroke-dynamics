# Paired pixel-space control results — 2026-08-21

**Protocol:** `docs/pixel-space-control-protocol.md`  
**Eligibility:** True  
**Status:** **Success**  
**Validation-selected family:** MLP  
**Latent Gate 2 decision:** Remains fail

## Configuration integrity

The completed run matched the frozen 1,000/200/300 paired split, three 100-example stress slices, model seeds `11`, `22`, and `33`, 30 epochs, and the frozen optimizer and architecture settings. The implementation reported:

- all counterfactual candidates unique;
- all metrics finite;
- overfit loss decreased by 87.2%;
- exact-oracle retrieval 100%;
- exact-oracle maximum action-region MSE `3.55e-15`;
- implementation sanity passed.

The selected MLP contains only 833 trainable parameters.

## Primary pixel prediction result

| Method | Test action-region MSE | Improvement vs identity |
|---|---:|---:|
| Identity | 0.499396 | — |
| Mean delta | 0.475834 | 4.7% |
| Selected MLP | **0.000249** | **99.950%** |
| Exact oracle | effectively zero | 100% |

The MLP also improved by **99.948%** versus mean delta.

## Four-way exact-action retrieval

| Method | Mean top-1 retrieval |
|---|---:|
| Identity | 3.67% |
| Mean delta | 2.67% |
| Linear | 96.11% |
| Selected MLP | **100%** |
| Exact oracle | 100% |

Every MLP seed retrieved the true result on all 300 test examples. The family seed standard deviation was zero. Its candidate-selection distribution was 100% true, and all three pairwise true-outcome win rates—position, width, and intensity—were 100%.

This directly contrasts with latent Gate 2, where the selected MLP achieved 27.7% four-way retrieval and only 40.7% true-vs-width pairwise accuracy on the same deterministic test distribution.

## Seed stability

| MLP seed | Test action-region MSE | Retrieval |
|---:|---:|---:|
| 11 | 0.000178 | 100% |
| 22 | 0.000376 | 100% |
| 33 | 0.000193 | 100% |

Seed 22 has somewhat higher absolute error but remains orders of magnitude better than trivial baselines and preserves perfect action retrieval. There is no seed collapse.

## Crowding

Selected-family improvement versus identity remained effectively complete:

- crowding 0: 99.973%;
- crowding 5: 99.955%;
- crowding 15: 99.907%.

The tiny degradation with crowding does not affect action discrimination.

## Stress slices

Mean MLP action-region MSE across three seeds was approximately:

- unseen width 5: `0.000246`;
- unseen intensities: `0.000557`;
- unseen crowding 10: `0.000275`.

These remain better than the corresponding identity baselines by roughly 99.9%. The unseen-intensity slice is the hardest, but its error remains very small.

## Linear result

The shared linear predictor achieved 96.1% retrieval despite substantially worse image reconstruction than the MLP. This shows that correct candidate ranking can coexist with nontrivial pixel error—the reverse of latent Gate 2, where low average latent error coexisted with poor ranking. The nonlinear MLP is needed for near-exact compositing, but the full-resolution pixel formulation already makes action identity highly recoverable to the linear model.

## Interpretation

The control succeeds decisively:

> Exact action information is recoverable by a tiny deterministic predictor in the full-resolution pixel formulation, including exact stroke width.

Therefore, the latent failure is not explained by broken data generation, ambiguous counterfactuals, insufficient action information, general inability to learn the deterministic transition, or unlucky initialization. The bottleneck lies in the **overall latent patch formulation** used here: frozen DINOv2 patch targets, coarse spatial tokenization/action coverage, normalized token loss geometry, or their interaction with the shared patch-wise predictor.

This control does **not** prove that DINOv2 alone is defective. Gate 1 showed that its patch features notice stroke changes, and the control changes both target space and spatial granularity. The defensible claim is that the tested latent formulation lacks the exact action precision available in the tested pixel formulation.

## Thesis consequence

The experimental core now supports a coherent three-part answer:

1. frozen DINOv2 spatial features preserve single-stroke changes;
2. a small latent predictor achieves strong average one-step error reduction but fails exact candidate discrimination, especially width;
3. a matched explanatory pixel formulation recovers exact actions perfectly.

No latent rerun or Gate 3 planning is needed for this bachelor’s thesis result. Contrastive latent losses, higher-resolution features, spatially interacting predictors, and width-aware objectives belong in future work or clearly labeled post-formal ablations.
