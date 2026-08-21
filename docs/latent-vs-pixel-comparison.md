# Latent versus pixel one-step dynamics comparison

## Purpose

This comparison explains why strong average latent prediction did not support exact action ranking, while a small full-resolution pixel predictor did. The comparison is explanatory rather than a claim that every possible latent method is inferior to pixel prediction.

## Shared experimental structure

Both experiments used:

- the same deterministic 64×64 grayscale line renderer;
- current canvases with 0, 5, or 15 prior strokes;
- the same action family of endpoints, width, and intensity;
- the same paired 1,000/200/300 train, validation, and test split seeds;
- the same model initialization seeds `11`, `22`, and `33`;
- residual prediction;
- identity and training mean-delta baselines;
- linear and small nonlinear learned predictors;
- validation-only family selection;
- balanced action-region and outside-region MSE;
- the same four unique retrieval classes: true, shifted position, changed width, and changed intensity.

## Important formulation differences

| Component | Latent experiment | Pixel control |
|---|---|---|
| Prediction target | Frozen DINOv2 final-layer patch-token residual | Normalized next-canvas pixel residual |
| Spatial resolution | 16×16 patch grid | 64×64 pixels |
| Proposed-action mask | Fractional patch coverage | Exact binary pixel mask |
| Current state input | 384-dimensional frozen patch token | One normalized pixel value |
| Selected MLP size | 199,808 parameters | 833 parameters |
| Retrieval distance | Normalized patch-feature MSE | Normalized pixel MSE |

Because these components change together, the comparison localizes the problem to the overall tested latent patch formulation. It does not isolate DINOv2 as the sole cause.

## Primary comparison

| Result | Latent MLP | Pixel MLP |
|---|---:|---:|
| Improvement vs identity | 61.8% | 99.950% |
| Improvement vs mean delta | 57.1% | 99.948% |
| Retrieval | 27.7% | 100% |
| True beats shifted position | 77.9% | 100% |
| True beats changed width | 40.7% | 100% |
| True beats changed intensity | 75.2% | 100% |
| Seed retrieval range | 26.3–29.3% | 100–100% |

Raw MSE values must not be compared across target spaces because latent and pixel errors use different units, dimensionalities, and normalization. Relative improvement within each target space and the shared four-way retrieval task are the meaningful comparisons.

## Crowding comparison

| Prior strokes | Latent improvement vs identity | Pixel improvement vs identity |
|---:|---:|---:|
| 0 | 79.0% | 99.973% |
| 5 | 43.3% | 99.955% |
| 15 | 25.0% | 99.907% |

Latent average-error performance degrades substantially with crowding but remains positive. Pixel performance remains near complete. More importantly, the latent model cannot preserve exact action identity even when its average prediction is useful.

## What the controls rule out

The paired pixel success makes the following explanations unlikely:

- broken transition generation;
- duplicate or ambiguous retrieval candidates;
- missing width or intensity information in the proposed action;
- insufficient training examples for the basic deterministic transition;
- general inability of a small deterministic predictor to learn the task;
- unlucky learned-model initialization;
- a broken retrieval implementation.

The exact compositing oracle’s effectively zero error and 100% retrieval independently validate the image-space evaluation path.

## What remains unresolved

The experiment does not distinguish among:

- information loss or invariance in frozen DINOv2 patch features;
- coarse 16×16 spatial tokenization;
- loss of thin-width detail in fractional patch action coverage;
- normalized token-MSE geometry;
- independent patch-wise prediction without spatial interaction;
- deterministic regression smoothing nearby latent outcomes.

These are mechanisms for future work, not reasons to revise the completed result.

## Main thesis claim

> In a controlled deterministic stroke-rendering task, frozen DINOv2 patch features preserved and supported average prediction of one-stroke changes, but low latent MSE did not provide the precision needed to rank closely related actions. A tiny full-resolution pixel predictor recovered the exact action for every paired test example, localizing the failure to the tested latent patch formulation rather than to the underlying transition’s learnability.

## Scope limits

This claim is restricted to synthetic grayscale straight-line strokes, 64×64 canvases, the tested frozen DINOv2 layer, the chosen action encoding, and small shared predictors. It is not evidence that all latent models fail, that DINOv2 is universally unsuitable, or that the pixel model is a complete painting agent.
