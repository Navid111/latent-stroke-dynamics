# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Controlled latent-planner runner implementation  
**Status:** Smoke complete and closed; controlled comparison unauthorized

## Formal result carried forward

- ranking-aware retrieval: 74.44%;
- MSE-only retrieval: 31.44%;
- gain: 43.00 points;
- formal integrity: passed;
- formal experiment: permanently closed.

## Smoke result

The one authorized five-method smoke passed implementation integrity. Final MSE was 0.138367 random, 0.084314 exact pixel, 0.092628 learned pixel, 0.095354 latent MSE, and 0.116399 latent ranking. Latent MSE showed stronger score-to-exact association and a better short trajectory than latent ranking on this diagnostic target. This cannot select or retune methods.

All learned replays were deterministic, target encoding and observed-canvas re-encoding checks passed, predicted latents were never rolled forward, hashes matched, and no training occurred. The smoke is closed and must not be rerun.

## Authorization boundary

The six-target controlled comparison remains unauthorized. No controlled target may be generated. No model training or fine-tuning is authorized.

## Next action

Implement a guarded five-method controlled runner with atomic output, overwrite refusal, frozen criteria, integrity checks, aggregation, and a validation-only unauthorized path. Then validate it without loading models or generating controlled data.
