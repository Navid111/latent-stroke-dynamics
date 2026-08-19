# Current State

**Last updated:** 2026-08-19  
**Branch:** `main`  
**Current gate:** Gate 1 — frozen-encoder stroke sensitivity  
**Gate status:** Initial smoke promising but inconclusive; version-2 diagnostic ready for local verification

## Objective

Determine whether frozen spatial visual features reliably preserve the local change caused by one controlled stroke. This evidence is required before training an action-conditioned next-representation predictor.

## Completed

- Created the private `latent-stroke-dynamics` repository.
- Implemented the deterministic renderer, frozen encoder wrapper, tests, and initial Gate 1 experiment.
- Fixed the missing `torchvision` dependency and Matplotlib compatibility issue.
- Successfully ran the first three-sample smoke test on a base-model M1 MacBook Air using CPU.
- Curated the first result snapshot under `results/gate1-smoke/2026-08-19/`.
- Recorded that blank-canvas patch features detect and approximately localize an added stroke.
- Recorded that moderate crowding, distributed noise sensitivity, and all-patch averaging remain unresolved.
- Implemented Gate 1 version 2 with paired crowding, a pixel-matched noise control, top-10% patch metrics, quantitative localization, faceted plots, and tests.
- Updated the Gate 1 protocol with criteria frozen before the 25-sample run.

## Empirical status

The initial smoke test is encouraging but not decisive:

- no-change behavior is numerically stable,
- added strokes on blank canvases produce strong, consistent feature changes,
- the qualitative patch heatmap follows the changed stroke,
- the global token is weak for position changes,
- add-stroke separation becomes inconsistent with five prior strokes,
- and tiny distributed noise produces unexpectedly large feature changes.

Gate 1 has neither passed nor failed. Version 2 must be validated locally and then run with 25 paired samples.

## Current decisions

- Use 64×64 grayscale canvases and one straight-line primitive.
- Use a fixed black, two-pixel-wide test stroke for the primary add-stroke comparison.
- Reuse each test stroke across nested crowding levels.
- Keep `facebook/dinov2-small` frozen as the first engineering baseline.
- Compare global, all-patch, top-10% patch, and changed-region metrics.
- Use fair noise controls and quantitative localization.
- Continue using CPU with batch size 4 on the M1 for now.
- Do not train a dynamics predictor until Gate 1 is evaluated.

## Next actions

1. Pull the version-2 code and documentation from `main`.
2. Run `pytest`; all renderer and Gate 1 tests should pass.
3. Run the three-sample version-2 smoke test at crowding 0 and 5.
4. Confirm that all expected CSVs and figures are generated.
5. Inspect separation, localization, and paired crowding behavior.
6. Fix only genuine implementation or design errors revealed by that smoke test.
7. If version 2 is structurally valid, run 25 paired samples at crowding 0, 5, and 15.
8. Record the full result and make an explicit Gate 1 decision.

## Immediate commands

```bash
git pull
pytest
python experiments/01_embedding_sensitivity.py \
  --samples 3 \
  --crowding 0 5 \
  --batch-size 4 \
  --device cpu \
  --output-dir outputs/gate1-v2-smoke
```

## Expected version-2 artifacts

- `results.csv`
- `aggregate_summary.csv`
- `distance_distributions.png`
- `localization_metrics.png`
- `example_patch_heatmap_crowding_0.png`
- `example_patch_heatmap_crowding_5.png`
- `run_config.json`

## Current blockers and risks

- Version 2 has been committed but not yet executed on the local environment.
- Python 3.14 remains newer than many research stacks, although the current inference path works.
- DINOv2 may prefer distributed texture changes over thin geometric marks.
- The practical gate criteria are project-specific and must be reported transparently.

## Handoff note

The next agent should verify Gate 1 version 2 and interpret the paired results. It should not begin the predictor, planner, reinforcement learning, complex brushes, or multi-step rollout unless this file is updated with evidence that Gate 1 passed.
