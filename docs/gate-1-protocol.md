# Gate 1 protocol — does the frozen encoder notice a stroke?

## Purpose

Before training an action-conditioned predictor, establish that the target representation contains usable information about a stroke's visual consequence. A dynamics model cannot reliably predict information that the frozen encoder discards.

This is a **representation diagnostic**, not yet the thesis's final benchmark.

## Pilot history

### Initial smoke test

The first three-sample run validated the pipeline and produced an encouraging blank-canvas heatmap. It also exposed weaknesses: different actions were used at different crowding levels, all-patch averaging diluted thin local changes, the plot pooled crowding levels, and localization was only visual.

### Version-2 smoke test

Version 2 used paired strokes across crowding, separated plots, dense pixel-MAE-matched noise, top-10% patch metrics, and quantitative localization.

Findings:

- no-change distances were numerical zero,
- added strokes localized strongly on blank and five-stroke canvases,
- median add-stroke localization lift was about `15.9×` at crowding 0 and `7.9×` at crowding 5,
- the global token weakened sharply under crowding while spatial localization remained strong,
- add-stroke top-10% distance beat dense matched noise in all three blank samples but none of the three five-stroke samples.

The dense control matched total pixel MAE but changed roughly half the canvas, while the stroke changed only about 1–2% of pixels. It was therefore retained as a stress test rather than used as the primary coherent-stroke comparison.

No Gate 1 decision was made from either pilot. Pilots are used only to debug and freeze the formal design.

## Final pre-run controlled design

For each `sample_id`:

1. Sample one canonical black, two-pixel-wide test stroke.
2. Generate one sequence of prior strokes.
3. Construct nested canvases with 0, 5, and 15 prior strokes from that same sequence.
4. Apply the same test stroke and its controlled variants to every crowding level.
5. Evaluate every comparison inside the same canonical reference-stroke region as well as globally.

This pairing makes crowding the intended context variable rather than confounding it with different stroke geometry, width, or intensity.

## Controlled comparisons

- `no_change` — identical image pair.
- `tiny_pixel_noise` — low-amplitude distributed noise.
- `pixel_matched_noise` — dense distributed noise with approximately the same pixel MAE as the added stroke; treated as a stress test.
- `sparse_pixel_matched_noise` — random non-line changes matching both the exact number of changed pixels and total absolute pixel difference of the added stroke; primary nuisance control.
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
- Mean distance inside the canonical reference-stroke region for every condition.

The top-10% metric avoids diluting a thin local change across all 256 patches. The reference-region metric compares the coherent stroke and nuisance controls at the location where the action actually occurred.

### Spatial localization

The exact pixel-change mask is downsampled to the encoder's patch grid. The script records:

- mean feature distance inside changed patches,
- mean feature distance outside changed patches,
- changed-region enrichment,
- top-k localization recall,
- lift over recall expected from random patch selection.

A lift above `1` means the largest feature changes overlap the true changed region more than random selection would. Inspect heatmaps as well; one scalar cannot reveal every failure mode.

### Crowding robustness

All plots separate crowding levels. Report blank, moderately occupied, and crowded canvases even if performance deteriorates.

### Global versus spatial features

A global embedding may recognize that the image remains “a drawing” while discarding where a stroke moved. Position-sensitive planning therefore requires spatial patch or intermediate features.

## Frozen engineering gate for the 25-sample run

These criteria are frozen before the formal run and must not be silently weakened afterward:

1. `no_change` distances remain at numerical zero.
2. `add_stroke` top-10% patch distance exceeds the paired `sparse_pixel_matched_noise` value in at least 80% of samples at crowding 0 and at least 70% at crowding 5.
3. `add_stroke` reference-region distance exceeds the paired sparse-control value in at least 80% of samples at crowding 0 and at least 70% at crowding 5.
4. Median localization top-k lift for `add_stroke` is at least 2.0 at crowding 0 and at least 1.5 at crowding 5.
5. Heatmaps and changed-region metrics agree that the response is concentrated around the stroke rather than only changing globally.
6. Dense pixel-matched noise is reported as a robustness stress test but is not the primary gate control because its spatial support is intentionally different.
7. Results at crowding 15 are reported as a stress test; failure there alone does not invalidate the small-scope thesis.

These are practical project gates, not universal scientific constants. Report raw paired percentages and distributions alongside the decision.

## Decision

### Pass

Proceed to a deterministic one-step predictor if controlled stroke changes are distinguishable from the sparse matched control, spatial changes are localized, and useful sensitivity remains at moderate crowding.

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
