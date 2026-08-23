# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Latent-space planner smoke validation  
**Status:** Five-method smoke runner implemented; smoke unauthorized

## Formal result carried forward

- ranking-aware retrieval: 74.44%;
- MSE-only retrieval: 31.44%;
- gain: 43.00 points;
- formal integrity: passed;
- formal experiment: permanently closed.

## Frozen latent planner

The planner compares random, exact pixel, learned pixel, three-seed MSE-only latent, and three-seed ranking-aware latent methods. Latent candidates are scored by full-grid normalized-feature MSE to the frozen target latent, averaged across seeds 11/22/33. Every selected stroke is executed exactly and the observed canvas is re-encoded.

Foundation validation passed with 79 tests. All seven loaded models were frozen, repeated encoding and scoring were deterministic, scores were finite, no planner data were generated, and no model was trained or fine-tuned. The six formal latent predictor state hashes are committed in `configs/latent-planner-2026-08-23.json`.

## Smoke runner

The guarded runner and unit tests are implemented. Validation-only mode checks the frozen five-method configuration, six hashes, authorization flags, and output-path availability without loading a model, generating the reserved target, or creating an output directory. The execution path preserves incomplete output, performs exact execution/re-encoding, saves best and final frames, and checks deterministic replay for all three learned methods.

## Authorization boundary

Smoke and controlled phases remain unauthorized. Do not generate target seed `20261201` or any controlled target. Do not train or fine-tune any model.

## Next action

Pull and run the full test suite, then run `python experiments/16_latent_planner_smoke.py --validate-only`. Review the unauthorized JSON before any separate smoke authorization commit.
