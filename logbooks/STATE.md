# Current State

**Last updated:** 2026-08-23  
**Branch:** `phase-b/saliency-latent`  
**Current stage:** Phase B0 protocol frozen; validation-only implementation next  
**Status:** Phase B0 implementation and every experimental data phase unauthorized

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled multi-step comparison remains a frozen criterion failure with implementation integrity passed: latent ranking mean final MSE was 1.996× exact pixel.

The score audit selected the MSE-only ensemble with normalized-latent L1. Forced L1 then reduced mean final MSE by about 7.05% versus forced latent MSE on all three planner-development targets. The zero-margin no-op stopped after only 3.33 strokes on average. The Stage A decision is permanently `not_eligible`, and its confirmatory phase remains prohibited.

## Phase B0 frozen direction

A new protocol now isolates a stronger latent model before saliency, color, or renderer changes. The fixed B0 system uses:

- the unchanged 64×64 grayscale straight-stroke renderer;
- a trainable multi-scale online encoder with 32×32×32 and 16×16×64 spatial latents;
- a momentum target encoder with stop-gradient targets;
- a two-channel pre-rendered stroke action;
- a multi-scale latent residual predictor;
- variance/covariance anti-collapse regularization;
- an explicit no-op transition;
- a target-conditioned progress head trained on exact pixel-MSE reduction;
- fixed progress regression and candidate-ranking terms.

All B0 development and formal seeds are new and disjoint. Development must pass eleven frozen eligibility conditions before formal B0 can be considered. B1 background→object→detail scheduling and B2 RGB/high-resolution painting each require a later separate protocol.

## Next action

Implement validation-only configuration, architecture, objective, EMA, gradient, parameter-count, and dummy-overfit checks. This stage may not load historical checkpoints, generate renderer data, create output directories, or authorize training. After the complete test suite passes on Navid's machine, archive the implementation manifest and decide whether to authorize exactly one B0 development execution.
