# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

## Current status

- **Gate 1 passed** on 2026-08-19: frozen DINOv2-small patch features preserved controlled stroke changes spatially.
- **Gate 2 formally failed** on 2026-08-21 under its frozen conjunctive rule.

Gate 2 produced a strong mixed result. The validation-selected three-seed MLP reduced test action-region MSE by **61.8% versus identity** and **57.1% versus mean delta**, remained positive at every crowding level, passed all sanity checks, and generalized to every stress slice. Exact counterfactual retrieval was only **27.7%**, below both the frozen 50% requirement and 35% fail boundary.

The conclusion is not that nothing was learned. Broad latent stroke consequences are highly predictable, but the current deterministic MSE predictors do not preserve enough exact action detail for planning.

See:

- [`docs/gate-2-results.md`](docs/gate-2-results.md)
- [`results/gate2-formal/2026-08-21/`](results/gate2-formal/2026-08-21/)
- [`docs/gate-2-protocol.md`](docs/gate-2-protocol.md)

## Frozen decision

| Criterion | Formal result | Outcome |
|---|---:|---|
| Improvement vs identity | 61.8% | Pass |
| Improvement vs mean delta | 57.1% | Pass |
| Positive at crowding 0/5/15 | Yes | Pass |
| Counterfactual retrieval | 27.7% | **Fail** |
| Sanity and seed stability | Passed | Pass |

The formal latent run must not be rerun or retuned. Gate 3 target-guided planning does not begin with this predictor.

## Next work

1. Run the existing no-retraining retrieval decomposition on the formal output for interpretation only.
2. Complete the formal artifact archive.
3. Implement the preregistered small action-conditioned pixel-space control.
4. Compare pixel and latent prediction for the thesis conclusion.

Potential contrastive losses, spatially interacting predictors, or width-specific objectives are future work or post-formal ablations and cannot replace the recorded Gate 2 result.
