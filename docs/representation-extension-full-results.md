# Full representation extension — raw result review

**Run:** single authorized frozen execution, 2026-08-22  
**Elapsed:** 2,353.79 seconds (39.23 minutes)  
**Rerun/retuning:** forbidden  
**Historical Gate 2, pixel-control, and Stage 3 decisions:** unchanged

## Important raw-run reporting issue

The raw summary says `full_extension_completed_with_failed_integrity_checks`. That red status comes from one extra runner guard, not a failed oracle ranking:

- task-autoencoder exact-target oracle top-1 was **100%**;
- all four encoded candidates were unique;
- separately batched encodings of candidate zero differed by at most `1.7404556274414062e-05`;
- the runner required that last value to be exactly zero.

The frozen written protocol requires oracle retrieval to equal 100% and separately requires unique encoded candidates. It never requires bit-identical candidate-zero encodings across separate batches. A pure summary-only adjudicator therefore preserves the raw output while applying the written protocol without retraining or metric recomputation.

A second raw-run classification mismatch is also adjudicated: Section 9 states that retrieval at or below 35% is `not_predictively_usable`, while the runner gave ViT-MAE the average-predictable label solely because its average errors passed.

## Autoencoder representation learning

The selected task autoencoder was seed 101 at epoch 50.

- validation reconstruction MSE: **0.00230759**;
- validation mean-image baseline MSE: **0.05092760**;
- improvement over that baseline: **95.47%**;
- test reconstruction MSE: **0.00266099**;
- test reconstruction MAE: **0.0155805**;
- mean train latent-channel standard deviation: **0.970844**;
- checkpoint reload maximum difference: **0.0**;
- total parameters: **49,569**.

All frozen autoencoder eligibility checks passed. The development reconstruction failure did not repeat with the larger frozen training split.

## Task-autoencoder dynamics

Selected family: MLP.

- action-region MSE improvement versus identity: **70.65%**;
- improvement versus mean delta: **68.62%**;
- four-way retrieval: **37.89%**;
- retrieval seed standard deviation: **2.80 percentage points**;
- true beat position alternative: **99.11%**;
- true beat width alternative: **68.89%**;
- true beat intensity alternative: **52.00%**;
- every model seed beat identity;
- all primary crowding levels improved.

Primary crowding improvements were 73.69% at crowding 0, 70.20% at crowding 5, and 65.99% at crowding 15. Secondary stress improvements remained positive: 76.78% for unseen width 5, 70.47% for unseen intensities, 58.49% at crowding 30, and 48.87% at crowding 60.

**Written-protocol classification:** `average_predictable_but_not_action_usable`.

This is the strongest compressed latent tested. It models average stroke consequences well and preserves position especially strongly, but it misses the 50% four-way action-retrieval requirement. Intensity is the main pairwise weakness.

## Frozen ViT-MAE dynamics

Selected family: MLP.

- action-region MSE improvement versus identity: **33.08%**;
- improvement versus mean delta: **30.63%**;
- four-way retrieval: **7.11%**;
- retrieval seed standard deviation: **0.19 percentage points**;
- true beat position alternative: **87.00%**;
- true beat width alternative: **25.00%**;
- true beat intensity alternative: **36.22%**.

Primary crowding improvements remained positive but weakened from 40.33% to 25.45%. Secondary crowding-60 performance was **13.69% worse than identity**.

**Written-protocol classification:** `not_predictively_usable`, because four-way retrieval is at or below the frozen 35% boundary.

The pretrained representation retains some position signal but is unsuitable for precise width/intensity action ranking in this setup.

## Representation ladder

Historical and new results are descriptive because the historical experiments used earlier paired seeds.

| Representation | Four-way retrieval | Interpretation |
|---|---:|---|
| Raw pixels | 100.00% | Full-resolution action identity is recoverable |
| Task autoencoder | 37.89% | Best tested latent; average-predictable, not action-usable |
| Frozen DINOv2-small | 27.67% | Historical formal latent Gate 2 fail |
| Frozen ViT-MAE | 7.11% | Not predictively usable for exact action ranking |

## Thesis-level conclusion

The extension narrows the explanation. Generic frozen visual representations were not precise enough for exact stroke-action ranking. Training a compact spatial representation on the project's own renderer canvases greatly improved reconstruction and latent transition prediction, and it outperformed both frozen pretrained latent anchors in retrieval. It still did not reach the frozen action-usable threshold, while raw pixels remained fully discriminative.

The honest claim is therefore not that latent prediction is impossible. It is that **representation choice and task alignment substantially change predictability, but the tested compressed latents still lose stroke-level action identity needed for reliable planning**. The successful painter should remain pixel-based.
