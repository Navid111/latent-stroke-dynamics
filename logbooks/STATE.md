# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Strong pilot localization; final sparse-control smoke test pending

## Objective

Determine whether frozen spatial visual features reliably preserve the local change caused by one controlled stroke. This evidence is required before training an action-conditioned next-representation predictor.

## Completed

- Implemented the deterministic renderer, frozen encoder wrapper, tests, and Gate 1 experiment.
- Fixed environment and plotting compatibility issues.
- Ran and archived the initial three-sample smoke test.
- Ran Gate 1 version 2 with paired actions across blank and five-stroke canvases; all six tests passed.
- Confirmed strong spatial localization of the added stroke on blank and moderately occupied canvases.
- Confirmed that a global token is inadequate for position-sensitive stroke changes.
- Diagnosed dense pixel-matched noise as a useful stress test but an unfair primary control because it changes nearly the entire canvas.
- Added a sparse nuisance control that exactly matches both changed-pixel count and total pixel difference.
- Added canonical reference-stroke-region metrics and automatic paired gate diagnostics.
- Froze the formal Gate 1 protocol before the 25-sample run.

## Empirical status

The version-2 pilot supports the feasibility of frozen spatial features:

- added-stroke localization lift was far above random at both crowding levels,
- crowded heatmaps remained aligned with the new stroke,
- changed-region response remained enriched relative to unchanged regions,
- absolute response magnitude weakened with crowding,
- dense distributed noise produced large feature changes.

Gate 1 has neither passed nor failed because only three samples were used and the final sparse matched control has not yet been executed locally.

## Current decisions

- Use 64×64 grayscale canvases and one straight-line primitive.
- Use a fixed black, two-pixel-wide test stroke.
- Reuse each test stroke across nested crowding levels.
- Keep `facebook/dinov2-small` frozen as the first engineering baseline.
- Use the sparse support-and-MAE-matched control as the primary nuisance comparison.
- Keep dense matched noise as a robustness stress test.
- Compare global, all-patch, top-10%, changed-region, and reference-region metrics.
- Continue using CPU with batch size 4 on the M1.
- Do not train a dynamics predictor until the formal Gate 1 result is evaluated.

## Next actions

1. Pull the final sparse-control implementation from `main`.
2. Run `pytest`; seven tests should pass.
3. Run the three-sample `gate1-v3-smoke` experiment at crowding 0 and 5.
4. Verify `gate_diagnostics.csv`, sparse-control changed-pixel matching, plots, and heatmaps.
5. Make no further design changes unless the smoke test reveals an implementation error.
6. If structurally valid, run the frozen 25-sample formal experiment at crowding 0, 5, and 15.
7. Archive the formal run and make the explicit Gate 1 decision.

## Immediate commands

```bash
git pull
pytest
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-v3-smoke
```

## Expected final-smoke artifacts

- `results.csv`
- `aggregate_summary.csv`
- `gate_diagnostics.csv`
- `distance_distributions.png`
- `localization_metrics.png`
- `example_patch_heatmap_crowding_0.png`
- `example_patch_heatmap_crowding_5.png`
- `run_config.json`

## Current blockers and risks

- The sparse-control implementation has not yet been run locally.
- Python 3.14 remains newer than many research stacks, although the current inference path works.
- DINOv2 may remain sensitive to dense distributed texture changes; this is now separated from the primary gate decision.
- The formal thresholds are project-specific and must be reported transparently.

## Handoff note

The next agent should verify the final sparse-control smoke test and then freeze the formal run. It should not begin the predictor, planner, reinforcement learning, complex brushes, or multi-step rollout unless Gate 1 is explicitly evaluated.
