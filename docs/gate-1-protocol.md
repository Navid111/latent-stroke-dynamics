# Gate 1 protocol — does the frozen encoder notice a stroke?

## Purpose

Before training an action-conditioned predictor, establish that the target representation contains usable information about a stroke's visual consequence. A dynamics model cannot reliably predict information that the frozen encoder discards.

This is a **representation diagnostic**, not yet the thesis's final benchmark.

## Pilot history

### Initial smoke test

The first run validated the pipeline and produced an encouraging blank-canvas heatmap. It also exposed weaknesses: different actions were used at different crowding levels, all-patch averaging diluted local changes, crowding was pooled in one plot, and localization was only visual.

### Paired dense-control pilot

The second design reused actions across crowding and added quantitative localization. Added-stroke localization was strong at both crowding levels, but dense pixel-MAE-matched noise changed roughly half the canvas while the stroke changed about 1–2% of pixels. Dense noise was retained as a stress test rather than the primary gate comparison.

### Final sparse-control pilot

The final three-sample pilot added random perturbations matching both the exact changed-pixel count and total absolute pixel difference of the coherent stroke.

The implementation checks passed:

- configuration contained `48 = 3 samples × 2 crowding levels × 8 conditions` pairs,
- sparse and coherent conditions matched changed-pixel count and pixel MAE per pair,
- no-change maximum patch distance remained below `4e-7`,
- the same canonical stroke was reused across crowding.

Pilot diagnostics:

| Crowding | Add > sparse in reference region | Median localization lift | Median reference enrichment |
|---:|---:|---:|---:|
| 0 | 3/3 | 12.40× | 2.05× |
| 5 | 3/3 | 11.52× | 5.54× |

The coherent stroke beat sparse noise on fixed top-10% patch distance in 3/3 blank samples but 0/3 five-stroke samples. The sparse control changes the same number of pixels but scatters them across roughly 20–27% of patch locations, whereas the connected line occupies roughly 4–7%. A fixed top-10% statistic therefore measures response to spatially dispersed high-frequency changes as much as action visibility. It remains a reported robustness statistic but is not the primary pass/fail criterion.

No pilot was used as the formal thesis result. The design below was frozen before the 25-sample run.

## Final controlled design

For each `sample_id`:

1. Sample one canonical black, two-pixel-wide test stroke.
2. Generate one sequence of prior strokes.
3. Construct nested canvases with 0, 5, and 15 prior strokes from that same sequence.
4. Apply the same test stroke and controlled variants at every crowding level.
5. Evaluate every condition in the same canonical reference-stroke region as well as globally.

## Controlled comparisons

- `no_change` — identical pair.
- `tiny_pixel_noise` — low-amplitude distributed noise.
- `pixel_matched_noise` — dense distributed noise with the same approximate pixel MAE; stress test.
- `sparse_pixel_matched_noise` — random non-line changes matching exact changed-pixel count and total absolute pixel difference; primary nuisance control.
- `add_stroke` — base versus base plus canonical stroke.
- `shift_position` — same stroke at nearby positions.
- `change_width` — one-pixel versus five-pixel width.
- `change_intensity` — dark versus light rendering.

## Metrics

### Primary

- Mean patch distance inside the canonical reference-stroke region.
- Paired add-stroke versus sparse-control reference-region win rate.
- Changed-region versus unchanged-region enrichment.
- Top-k localization recall and lift over random.
- Heatmap agreement with the exact stroke mask.

### Secondary and stress-test

- Global-token distance.
- Mean and maximum patch distance.
- Fixed top-10% patch distance.
- Tiny-noise response.
- Dense pixel-matched-noise response.

The secondary metrics remain important failure analyses, but a diffuse nuisance response does not by itself show that the coherent action is absent from the correct spatial state.

## Frozen engineering gate for the formal run

These criteria were frozen before viewing the 25-sample result:

1. `no_change` distances remain at numerical zero.
2. Add-stroke reference-region distance exceeds the paired sparse-control value in at least 80% of samples at crowding 0 and at least 70% at crowding 5.
3. Median add-stroke localization lift is at least 2.0 at crowding 0 and at least 1.5 at crowding 5.
4. Median add-stroke reference-region enrichment is greater than 1 at crowding 0 and 5.
5. Heatmaps and changed-region metrics agree that the response is concentrated around the stroke rather than only changing globally.
6. Fixed top-10%, global, tiny-noise, dense-noise, width, intensity, and position results are all reported, including failures, but are secondary diagnostics.
7. Crowding 15 is a stress test. Failure there alone does not invalidate the scoped Gate 1 result, but it must be discussed.

These are practical project gates, not universal scientific constants. Raw paired percentages and distributions accompany the decision.

## Formal result — 2026-08-19

The frozen run used seed `20260819`, 25 samples per crowding level, eight comparisons, and 600 total pairs. Artifacts are archived under `results/gate1-formal/2026-08-19/`.

| Crowding | Add > sparse top-10% | Add > sparse reference region | Median localization lift | Median reference enrichment | No-change max |
|---:|---:|---:|---:|---:|---:|
| 0 | 68% | 100% (25/25) | 12.80× | 2.05× | 2.98e-7 |
| 5 | 8% | 96% (24/25) | 10.24× | 4.95× | 3.58e-7 |
| 15 | 4% | 100% (25/25) | 10.60× | 6.77× | 3.58e-7 |

### Criterion evaluation

1. **Pass:** no-change remained at numerical zero at every crowding level.
2. **Pass:** paired reference-region win rates were 100% at crowding 0 and 96% at crowding 5, above the frozen 80% and 70% thresholds.
3. **Pass:** median localization lifts were 12.80× and 10.24×, far above the frozen 2.0× and 1.5× thresholds.
4. **Pass:** median reference-region enrichment was greater than one at both primary crowding levels.
5. **Pass:** changed-region localization statistics and archived heatmaps indicate a spatially concentrated response. Median localization lift remained above 10× at every crowding level.
6. **Reported:** fixed top-10% separation weakened sharply with crowding, as predicted from the sparse control's wider patch coverage. Global response also weakened with crowding. These are retained as limitations rather than used to overwrite the primary criteria.
7. **Stress result:** crowding 15 also passed the reference-region and localization checks, although it was not required for the scoped pass.

## Decision

**Gate 1 passes.** The representation contains usable local information about the visual consequence of a stroke under the scoped synthetic conditions. Proceed to a deterministic one-step predictor.

This does not yet show that a learned dynamics model can predict the next representation or rank candidate strokes; those are Gate 2 and Gate 3.

## What comes after the pass

1. Generate new `(canvas, action, next_canvas)` transitions with separate train, validation, and test seeds.
2. Keep the selected encoder frozen.
3. Train a small deterministic residual predictor over spatial patch features.
4. Compare against no-change, mean-delta, and linear baselines.
5. Test candidate-stroke ranking only after one-step prediction succeeds.
6. Treat depth-2 or depth-3 planning as optional.
