# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Pixel-space explanatory control  
**Status:** Protocol frozen; implementation committed; local tests and smoke pending

## Closed results

- Gate 1 formally passed.
- Latent Gate 2 formally failed because retrieval was 27.7%, despite strong and stable average-error prediction.
- Formal Gate 2 diagnosis isolated width as the main failure: 40.7% pairwise true-vs-width and 48.2% width-alternative selections.

No latent rerun or test tuning is authorized.

## Pixel-control implementation

Committed components:

- normalized grayscale pixel tensors;
- exact full-resolution proposed-action masks;
- current pixel, seven-value action, mask, and coordinate inputs;
- identity and training mean-delta baselines;
- shared affine and `11 -> 64 -> 1` MLP predictors;
- renderer-equivalent exact compositing oracle;
- balanced inside/outside residual loss;
- clamped next-canvas metrics;
- unique four-way pixel retrieval;
- seed-aware summaries, subgroup tables, and plots;
- paired-control eligibility guard;
- deterministic unit tests.

## Next actions

1. Pull the implementation and run all 24 tests locally.
2. Run the 128/32/64 development-only engineering smoke on seeds `20260830`–`20260832`.
3. Review sanity, oracle exactness, metrics, retrieval, and plots.
4. Only then run the single frozen paired control.

## Boundaries

- The smoke must remain `diagnostic_only`.
- Do not run the paired command before smoke review.
- The pixel control cannot revise the latent Gate 2 fail.
- Do not begin Gate 3 planning.
