# Gate 1 protocol — does the frozen encoder notice a stroke?

## Purpose

Before training an action-conditioned predictor, establish that the target representation contains usable information about a stroke's visual consequence. A dynamics model cannot reliably predict information that the frozen encoder discards.

This is a **representation diagnostic**, not yet the thesis's final benchmark.

## Initial smoke-test finding

The first three-sample run validated the pipeline and produced an encouraging blank-canvas heatmap: the strongest patch-feature changes followed the added line. It also exposed weaknesses in the first diagnostic design:

- different random actions were used at different crowding levels,
- averaging every patch diluted thin local changes,
- tiny distributed noise caused unexpectedly large feature distances,
- the original plot pooled crowding levels,
- localization was only visual.

No Gate 1 decision was made from that run. The archived snapshot is under `results/gate1-smoke/2026-08-19/`.

## Version 2 controlled design

For each `sample_id`:

1. Sample one canonical black, two-pixel-wide test stroke.
2. Generate one sequence of prior strokes.
3. Construct nested canvases with 0, 5, and 15 prior strokes from that same sequence.
4. Apply the same test stroke and its controlled variants to every crowding level.

This pairing makes crowding the intended context variable rather than confounding it with different stroke geometry, width, or intensity.

## Controlled comparisons

- `no_change` — identical image pair.
- `tiny_pixel_noise` — low-amplitude distributed noise.
- `pixel_matched_noise` — distributed noise with approximately the same pixel MAE as the added stroke.
- `add_stroke` — base canvas versus base plus the canonical stroke.
- `shift_position` — the same stroke at two nearby positions.
- `change_width` — one-pixel versus five-pixel width.
- `change_intensity` — dark versus light rendering of the same stroke.

Start with 64×64 grayscale images and one straight-line primitive. Complexity can be added only after this test is interpretable.

## Metrics

### Sanity and separation

- Pixel mean absolute difference.
- Global-token cosine distance.
- Mean patch-token cosine distance.
- Maximum patch-token cosine distance.
- Mean of the top 10% most-changing patch tokens.

The top-10% metric is included because a thin stroke should affect a small spatial subset; averaging all 256 patches can hide a useful local signal.

### Spatial localization

The exact pixel-change mask is downsampled to the encoder's patch grid. The script records:

- mean feature distance inside changed patches,
- mean feature distance outside changed patches,
- changed-region enrichment,
- top-k localization recall,
- lift over the recall expected from random patch selection.

A lift above `1` means the largest feature changes overlap the true changed region more than random selection would. Inspect the heatmaps as well; one scalar cannot reveal every failure mode.

### Crowding robustness

All plots separate crowding levels. Report blank, moderately occupied, and crowded canvases even if performance deteriorates.

### Global versus spatial features

A global embedding may recognize that the image remains “a drawing” while discarding where the stroke moved. Position-sensitive planning therefore requires spatial patch or intermediate features.

## Provisional engineering gate for the 25-sample run

These criteria are frozen before the larger run and must not be silently weakened afterward:

1. `no_change` distances remain at numerical zero.
2. For `add_stroke`, the top-10% patch distance exceeds the paired pixel-matched-noise value in at least 80% of samples at crowding 0 and at least 70% at crowding 5.
3. Median localization top-k lift for `add_stroke` is at least 2.0 at crowding 0 and at least 1.5 at crowding 5.
4. Heatmaps and changed-region metrics agree that the response is concentrated around the stroke rather than only changing globally.
5. Results at crowding 15 are reported as a stress test; failure there alone does not invalidate the small-scope thesis.

These are practical project gates, not universal scientific constants. Report the raw paired percentages and distributions alongside the decision.

## Decision

### Pass

Proceed to a deterministic one-step predictor if controlled stroke changes are distinguishable from fair controls, spatial changes are localized, and useful sensitivity remains at moderate crowding.

### Borderline

If localization is convincing but separation or crowding robustness is weak, try one justified change at a time:

1. an intermediate encoder layer,
2. a different frozen encoder,
3. a larger canvas or thicker primitive,
4. a feature normalization or spatial pooling choice.

Document every attempted change and do not tune on the final test set.

### Fail

If multiple sensible frozen-feature configurations cannot preserve one-stroke changes, do not force a world-model experiment. The result can become a scoped thesis about the suitability of frozen visual representations for incremental stroke-based rendering.

## What comes after a pass

1. Generate `(canvas, action, next_canvas)` transitions.
2. Freeze the selected encoder.
3. Train a small deterministic residual predictor for the next spatial representation.
4. Compare against no-change, mean-delta, and linear baselines.
5. Test whether predicted outcomes correctly rank candidate strokes.
6. Treat depth-2 or depth-3 planning as optional.
