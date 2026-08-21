# Pixel-space action-conditioned control protocol

**Status:** Frozen before implementation  
**Frozen on:** 2026-08-21  
**Relationship to Gate 2:** Explanatory control after the recorded formal Gate 2 fail; it cannot revise that decision

## 1. Question

Can the same class of small deterministic action-conditioned predictor recover exact one-stroke outcomes when it predicts normalized canvas pixels rather than frozen DINOv2 patch-token residuals?

The control distinguishes two explanations for the Gate 2 result:

1. **Latent-formulation bottleneck:** a pixel predictor recovers exact actions while the latent predictor does not.
2. **General predictor bottleneck:** pixel prediction also fails exact action discrimination.

The control is not Gate 3 planning and does not use a target image.

## 2. Provenance and limitation

Gate 2 required a small pixel-space predictor before the final thesis comparison. This detailed protocol is frozen after observing the formal latent result but before implementing or running the pixel model. It is therefore an explanatory paired control, not an independent preregistered confirmation.

The control deliberately reuses the exact Gate 2 transition distribution and paired split seeds so differences are not caused by easier canvases or actions. Because the pixel formulation uses full-resolution pixels and an exact full-resolution proposed-action mask, a successful result localizes the bottleneck to the overall latent patch formulation; it cannot by itself separate frozen target features, patch resolution, and latent loss geometry.

## 3. Frozen data

Reuse the deterministic Gate 2 transition generator and paired formal specifications:

| Split | Examples | Seed |
|---|---:|---:|
| Train | 1,000 | `20260824` |
| Validation | 200 | `20260825` |
| Test | 300 | `20260826` |
| Unseen width 5 | 100 | `20260827` |
| Unseen intensities | 100 | `20260828` |
| Unseen crowding 10 | 100 | `20260829` |

Primary crowding is `{0, 5, 15}`, widths are `{1, 2, 3, 4}`, grayscale values are `{0, 32, 64, 96, 128}`, and minimum normalized stroke length is `0.20`.

An engineering smoke may use 128/32/64 examples from new development-only seeds `20260830`, `20260831`, and `20260832`. Smoke data cannot decide the control result.

## 4. Pixel representation

Convert grayscale pixels to floating point in `[0, 1]`, where 0 is black and 1 is white:

```text
p_current = C_current / 255
p_next    = C_next / 255
delta_p   = p_next - p_current
```

Predict a residual and reconstruct the next canvas:

```text
delta_hat_p = G(p_current, action)
p_hat_next  = clamp(p_current + delta_hat_p, 0, 1)
```

Training uses residual error. Evaluation and retrieval use the clamped predicted next canvas, fixed here before implementation.

## 5. Frozen action-conditioned inputs

For every full-resolution pixel, concatenate:

1. current normalized pixel value;
2. the same seven-value global stroke vector used in Gate 2;
3. an exact binary proposed-stroke mask rendered independently of canvas content;
4. normalized pixel-center x/y coordinates.

The resulting per-pixel input dimension is 11. The mask is available before committing the action and is not derived from the target next canvas.

## 6. Predictors and baselines

All remain visible:

1. **Identity:** predict zero residual.
2. **Mean delta:** repeat the training-set mean pixel residual image.
3. **Linear pixel predictor:** one shared affine map from the 11 per-pixel inputs to one residual value.
4. **Small MLP pixel predictor:** shared `11 -> 64 -> 1` network with GELU.
5. **Exact compositing oracle:** combine the current canvas, exact action mask, and stroke intensity analytically. This is a renderer-equivalent ceiling and implementation check, not a learned baseline.

Use learned-model initialization seeds `11`, `22`, and `33`. Select linear or MLP using mean validation action-region pixel MSE across its three seeds, never test retrieval.

## 7. Training

Frozen settings:

- objective: `0.5 * inside-action MSE + 0.5 * outside-action MSE` on pixel residuals;
- optimizer: AdamW;
- maximum epochs: 30;
- patience: 6;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- hidden dimension: 64;
- batch size: 16;
- device: CPU;
- model seeds: `11`, `22`, `33`.

A four-example, 30-step overfit check at learning rate `0.005` must reduce loss before the paired evaluation is interpreted.

## 8. Metrics

### Primary

- action-region clamped next-canvas MSE;
- relative improvement over identity and training mean delta;
- four-way pixel counterfactual top-1 retrieval.

### Secondary

- full-canvas and outside-region MSE;
- action-region MAE;
- results by crowding, stroke width, intensity, and length;
- three stress slices;
- seed spread and training curves;
- one residual/error visualization.

## 9. Counterfactual retrieval

Use the exact same four pixel-distinct rendered outcomes as latent Gate 2:

1. true stroke;
2. shifted position;
3. changed width;
4. changed intensity.

Score raw normalized-pixel MSE between the clamped prediction and each candidate over the union of all candidate action masks. Candidate zero is the true outcome; chance is 25%.

Report candidate-selection frequencies and pairwise true-versus-position, true-versus-width, and true-versus-intensity win rates. The exact compositing oracle must retrieve the true outcome on 100% of valid examples; otherwise the control implementation is invalid.

## 10. Frozen interpretation rule

The pixel control is called **successful** if the validation-selected learned family, averaged over three seeds:

1. improves action-region MSE by at least 30% versus identity and mean delta;
2. remains positive versus identity at crowding 0, 5, and 15;
3. reaches at least 50% four-way retrieval;
4. has no sanity failure or seed collapse.

Retrieval from 35% to below 50% is partial/inconclusive. Retrieval at or below 35% reproduces the action-discrimination failure.

These labels describe the control only. They cannot convert the recorded latent Gate 2 fail into a pass.

## 11. Interpretation matrix

| Pixel result | Interpretation |
|---|---|
| Pixel retrieval ≥50% and width pairwise >50% | Exact action information is recoverable in the pixel formulation; the latent patch formulation is the likely bottleneck |
| Pixel MSE strong but retrieval ≤35% | Deterministic average-error prediction again hides action-level failure |
| Pixel MSE and retrieval both weak | Small shared predictor or action encoding is insufficient even in image space |
| Oracle below 100% | Implementation invalid; repair before interpretation |

## 12. Required artifacts

Save configuration, split fingerprints, per-example and aggregate metrics, metrics by crowding, training history, retrieval rows, candidate preferences, pairwise diagnostics, overfit check, decision summary, and plots. Generated raw outputs remain ignored by Git; compact final summaries may be archived under `results/pixel-control/`.

## 13. Execution order

1. Commit this protocol before pixel-model code.
2. Implement pixel tensors, predictors, balanced loss, metrics, oracle, and tests.
3. Run local tests.
4. Run the development-only engineering smoke.
5. Repair implementation defects only; do not choose settings from smoke retrieval.
6. Run the paired control once with the frozen configuration.
7. Archive and compare against the already-recorded latent result.
