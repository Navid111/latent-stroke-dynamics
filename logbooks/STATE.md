# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current gate:** Gate 2 complete — formal fail  
**Gate status:** Closed under the frozen rule; fallback control next

## Formal Gate 2 decision

The exact frozen run completed with `formal_eligible: true` and selected the MLP family by validation error.

- action-region MSE: 0.000860;
- improvement versus identity: 61.8%;
- improvement versus mean delta: 57.1%;
- crowding improvements: +79.0%, +43.3%, and +25.0%;
- counterfactual retrieval: 27.7%;
- all seeds beat identity;
- overfit, finite-metric, and candidate-uniqueness checks passed.

The gate **failed** solely because retrieval was below the frozen 50% requirement and the 35% fail boundary. See `docs/gate-2-results.md`.

## Interpretation

The predictor robustly learns the broad one-step latent consequence of a stroke, including on unseen width, intensity, and crowding slices. It does not preserve enough exact action detail for reliable four-way counterfactual ranking. Average latent error and action-level usefulness are therefore not interchangeable.

This is a valid mixed thesis result, not a broken experiment.

## Completed

- Gate 1 formally passed and was archived.
- Gate 2 protocol, amendment, implementation, development diagnostics, and formal command were committed before the formal run.
- The formal run used untouched amended data and all three preregistered model seeds.
- The formal Gate 2 result was recorded without rerunning or post-test tuning.
- Decisive configuration and diagnostic artifacts were archived under `results/gate2-formal/2026-08-21/`.

## Next actions

1. Do not rerun or retune the completed formal latent experiment.
2. Run the no-retraining retrieval decomposition on the formal output for interpretation only.
3. Collect the remaining formal plots and training history for the archive.
4. Implement the preregistered small action-conditioned pixel-space control.
5. Compare pixel-space and latent behavior for the final thesis result.
6. Do not begin Gate 3 target-guided planning with the failed latent predictor.

## Immediate next step

Complete the formal artifact review, then scope the pixel-space control. Any contrastive loss, spatial architecture, or width-specific objective is post-formal future work and cannot replace the recorded result.
