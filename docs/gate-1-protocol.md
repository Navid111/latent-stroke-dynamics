# Gate 1 protocol — does the frozen encoder notice a stroke?

## Purpose

Before training an action-conditioned predictor, establish that the target representation contains usable information about the action's visual consequence. A dynamics model cannot reliably predict information that the frozen encoder discards.

This is a **representation diagnostic**, not yet the thesis's final benchmark.

## Controlled factors

Keep everything constant except one factor at a time:

- no change,
- tiny pixel noise,
- stroke presence,
- stroke position,
- stroke width,
- stroke intensity,
- number of existing strokes on the canvas.

Start with 64×64 grayscale images and a single straight-line primitive. Complexity can be added only after this test is interpretable.

## What to inspect

### 1. Separation

The distance distribution for meaningful stroke changes should be consistently separated from identical-canvas and tiny-noise controls. Do not judge this from a single attractive example.

A useful first practical criterion is that at least 90% of `add_stroke` patch-distance values exceed the 95th percentile of the tiny-noise control. Record the exact percentage rather than silently changing this criterion after seeing results.

### 2. Spatial localization

Patch-token difference heatmaps should concentrate around the changed stroke region. A global embedding can change while discarding *where* the stroke happened, which would be inadequate for spatial planning.

The initial script provides qualitative heatmaps. If they look promising, the next commit should add a quantitative localization score by comparing the top-changing feature patches with the rendered pixel-change mask.

### 3. Crowding robustness

Repeat the same intervention on blank, moderately occupied, and crowded canvases. The signal may weaken as the canvas fills. Plot the result; do not hide this failure mode by reporting only blank canvases.

### 4. Global versus spatial features

Compare the class/global token with patch tokens. For this thesis, spatial patch features are expected to be more useful for position-sensitive differences. A model that preserves only semantic identity—"this is still a drawing"—is not sufficient.

## Decision

### Pass

Proceed to a deterministic one-step predictor if:

- controlled changes are reliably distinguishable from controls,
- patch changes are spatially meaningful,
- and useful sensitivity remains on moderately crowded canvases.

### Borderline

Before abandoning the idea, try one change at a time:

1. an intermediate encoder layer,
2. a different frozen encoder,
3. a larger canvas or thicker primitive,
4. a feature normalization or spatial pooling choice.

Document every attempted change.

### Fail

If multiple sensible frozen-feature configurations cannot preserve one-stroke changes, do not force the world-model experiment. The result can become a scoped thesis about the suitability of frozen visual representations for incremental stroke-based rendering.

## What comes after a pass

1. Generate `(canvas, action, next_canvas)` transitions.
2. Freeze the selected encoder.
3. Train a small deterministic residual predictor for the next spatial representation.
4. Compare against "no representation change" and linear baselines.
5. Only then test whether predicted outcomes correctly rank candidate strokes.
6. Treat depth-2 or depth-3 planning as optional.
