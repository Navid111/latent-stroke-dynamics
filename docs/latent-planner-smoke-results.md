# Latent-planner implementation smoke result — 2026-08-23

**Status:** Engineering integrity passed  
**Role:** one-target diagnostic only  
**Rerun allowed:** No  
**Controlled comparison:** Unauthorized

## Integrity

- all 84 tests passed before execution;
- the one separately authorized smoke completed;
- all six latent predictor hashes and the pixel predictor hash matched;
- the target was encoded once per latent method;
- the exact observed canvas was re-encoded at every step;
- predicted latents were never rolled forward as state;
- learned pixel, latent MSE, and latent ranking replays were deterministic;
- no model was trained or fine-tuned;
- historical results remained unchanged.

## Outcome

| Method | Final MSE | Improvement from blank | Exact top-1 | Exact top-5 | Mean exact rank | Mean regret | Mean score–exact Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.138367 | 16.39% | — | — | — | — | — |
| Exact pixel | 0.084314 | 49.05% | — | — | — | — | — |
| Learned pixel | 0.092628 | 44.03% | 55% | 75% | 5.05 | 0.000862 | — |
| Latent MSE | 0.095354 | 42.38% | 25% | 90% | 3.95 | 0.001468 | 0.622 |
| Latent ranking | 0.116399 | 29.66% | 10% | 45% | 7.05 | 0.003136 | 0.414 |

All five methods reached their lowest MSE at step 20. The latent-MSE planner finished 31.09% below random and at 1.131× exact pixel. The latent-ranking planner finished 15.88% below random and at 1.381× exact pixel. Learned pixel was the strongest learned method on this target at 1.099× exact pixel.

## Interpretation

The smoke establishes implementation viability: both latent ensembles can drive a multi-step exact-execution painter, improve substantially from blank, and replay deterministically. Latent-MSE scores aligned more strongly with exact one-step pixel outcomes than ranking-aware scores on this target and produced the better trajectory.

The ranking-aware predictor's large formal one-step retrieval advantage did not translate into a ranking-planner advantage in this single short smoke. This is scientifically useful but not yet a controlled conclusion. The smoke is forbidden from selecting a different model, changing the score, or revising thresholds. In particular, applying the future controlled 20%-below-random criterion to this one target would give latent ranking only 15.88%; that observation cannot decide the six-target outcome.

## Decision

Implementation integrity passed, so it is permissible to implement and validate the guarded controlled runner. The controlled comparison remains unauthorized until that runner passes a separate no-data validation and receives a separate one-run authorization.
