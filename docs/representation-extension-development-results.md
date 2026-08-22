# Representation extension development smoke — 2026-08-22

**Status:** Complete  
**Decision-making:** No  
**Implementation integrity:** Passed  
**Primary/stress data generated:** No

## Purpose

This 128/32/64-transition run verifies the frozen post-core implementation before the untouched extension evaluation. Its metrics cannot classify either representation or change any previous result.

## Integrity

- All losses and metrics were finite.
- Every encoded counterfactual set remained unique.
- Exact-target oracle retrieval was 100% for both representations.
- Tiny-overfit loss decreased for both representations.
- Every dynamics seed beat identity.
- Autoencoder checkpoint reload reproduced identical features.
- Train-only latent statistics were non-collapsed.
- No primary or stress split was generated.

## Task-autoencoder development signal

| Metric | Development result |
|---|---:|
| Selected dynamics family | MLP |
| Action-region MSE improvement vs identity | 71.01% |
| Improvement vs mean delta | 66.31% |
| Four-way retrieval | 46.35% |
| Position pairwise win | 95.83% |
| Width pairwise win | 58.33% |
| Intensity pairwise win | 87.50% |
| Crowding improvements | 77.25%, 69.36%, 66.47% |

The task latent produced strong average-error prediction and approached, but did not reach, the frozen 50% retrieval threshold. Width remains the hardest action factor. Because this is development-only, the result is neither a pass nor a near-pass claim.

### Reconstruction caveat

The selected development autoencoder did not beat the train-mean-image validation baseline: reconstruction improvement was `-14.02%`, versus the frozen requirement of at least `+30%`. Its latent was non-collapsed and its dynamics signal was meaningful, but the untouched full run must enforce the reconstruction criterion. No architecture, loss, threshold, epoch budget, or seed is changed from this observation.

## Frozen ViT-MAE development signal

| Metric | Development result |
|---|---:|
| Selected dynamics family | Linear |
| Action-region MSE improvement vs identity | 23.97% |
| Improvement vs mean delta | 20.83% |
| Four-way retrieval | 7.29% |
| Position pairwise win | 79.17% |
| Width pairwise win | 21.88% |
| Intensity pairwise win | 29.69% |
| Crowding improvements | 29.50%, 22.53%, 20.97% |

ViT-MAE noticed enough transition structure for positive average-error improvement at every crowding level, but the action prediction selected the true encoded consequence much less often than the 25% random-choice reference. Position was comparatively preserved; width and intensity were severe confusions. The full untouched run remains required because smoke metrics are non-decision-making.

## Reporting correction

The archived raw smoke summary reports autoencoder `parameter_count: 0`. That field accidentally used the project's trainable-parameter helper after checkpoint loading had frozen every parameter. The architecture actually contains **49,569 total parameters**. We preserve the raw value and correct the reporting helper prospectively; no metric, model, selection, or result is affected.

## Next step

Repair only the parameter-count report, freeze the full command without changing scientific settings, run validation-only checks, and then execute the untouched primary/stress extension once.
