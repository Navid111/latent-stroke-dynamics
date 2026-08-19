# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Final pilot validated; formal 25-sample run ready

## Objective

Determine whether frozen spatial visual features reliably preserve the local change caused by one controlled stroke. This evidence is required before training an action-conditioned next-representation predictor.

## Completed

- Implemented deterministic rendering, frozen DINOv2 encoding, paired controls, plots, metrics, and tests.
- Fixed environment and plotting compatibility issues.
- Ran an initial implementation pilot.
- Ran a paired dense-noise pilot and confirmed strong spatial localization under blank and moderate crowding.
- Added a sparse nuisance control matching exact changed-pixel count and total pixel difference.
- Ran the final three-sample pilot with 48 paired comparisons.
- Verified exact sparse-control matching and numerical-zero no-change behavior.
- Confirmed add-stroke reference-region wins in 3/3 samples at crowding 0 and 3/3 at crowding 5.
- Confirmed median localization lift of 12.40× at crowding 0 and 11.52× at crowding 5.
- Froze the formal protocol and primary criteria before the 25-sample run.

## Empirical status

Pilot evidence strongly supports spatial stroke visibility:

- coherent strokes remain identifiable in the correct region under five-stroke clutter,
- localization and reference-region separation are strong,
- global-token response weakens under crowding,
- dense and sparse pixel noise can create larger diffuse responses than the coherent line on global or fixed top-10% metrics.

These are pilot findings, not the formal thesis result. Gate 1 has neither passed nor failed until the 25-sample run is evaluated.

## Frozen formal decisions

- 64×64 grayscale canvases.
- Fixed black, two-pixel-wide canonical stroke.
- Nested crowding levels 0, 5, and 15.
- Frozen `facebook/dinov2-small` patch representation.
- Sparse support-and-MAE-matched nuisance as primary control.
- Reference-region separation and localization as primary metrics.
- Global, fixed top-10%, tiny-noise, dense-noise, width, intensity, and position as required secondary diagnostics.
- CPU batch size 4 on the M1.
- No further design changes unless an implementation error invalidates execution.
- No predictor training until Gate 1 is explicitly evaluated.

## Next actions

1. Pull the final documentation update from `main`.
2. Optionally archive the final pilot as `results/gate1-v3-smoke/2026-08-19/`.
3. Run the frozen 25-sample experiment at crowding 0, 5, and 15.
4. Preserve `gate_diagnostics.csv`, full results, aggregate summary, configuration, and figures.
5. Make an explicit Gate 1 pass, borderline, or fail decision against the frozen criteria.
6. Only after a pass, begin the deterministic one-step predictor.

## Immediate commands

```bash
git pull
python experiments/01_embedding_sensitivity.py \
  --samples 25 \
  --crowding 0 5 15 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-formal
```

## Expected formal artifacts

- `results.csv`
- `aggregate_summary.csv`
- `gate_diagnostics.csv`
- `distance_distributions.png`
- `localization_metrics.png`
- `example_patch_heatmap_crowding_0.png`
- `example_patch_heatmap_crowding_5.png`
- `example_patch_heatmap_crowding_15.png`
- `run_config.json`

## Handoff note

The next agent should evaluate the frozen formal Gate 1 run. It should not change metrics after seeing results or begin the predictor unless the gate is explicitly passed.
