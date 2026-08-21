# Latent Stroke Dynamics

Bachelor's thesis experiments on **action-conditioned latent canvas dynamics for stroke-based rendering**.

## Current status

- **Gate 1 passed** on 2026-08-19.
- **Gate 2 formally failed** on 2026-08-21 under its frozen conjunctive rule.
- **Gate 2 formal diagnosis is complete.**

The validation-selected MLP reduced action-region latent MSE by **61.8% versus identity** and **57.1% versus mean delta**, remained positive at every crowding level, passed all sanity checks, and generalized to every stress slice. Exact retrieval was only **27.7%**.

The failure is specific rather than global. Across three stable seeds, true outcomes beat position and intensity alternatives 77.9% and 75.2% of the time, but beat width alternatives only 40.7%. The MLP selected width-changed outcomes 48.2% of the time.

The resulting thesis claim is:

> Frozen DINOv2 spatial features support accurate average one-step stroke dynamics, but low latent MSE does not guarantee precise discrimination among closely related stroke actions.

See:

- [`docs/gate-2-results.md`](docs/gate-2-results.md)
- [`docs/gate-2-formal-visual-review.md`](docs/gate-2-formal-visual-review.md)
- [`docs/gate-2-formal-retrieval-diagnostics.md`](docs/gate-2-formal-retrieval-diagnostics.md)
- [`results/gate2-formal/2026-08-21/`](results/gate2-formal/2026-08-21/)

## Next experiment

The next task is the preregistered minimal **action-conditioned pixel-space control**. It will test whether precise width discrimination is recoverable when predicting canvas pixels rather than frozen latent patch-token residuals.

- If pixel retrieval succeeds, the frozen latent target/objective is the likely bottleneck.
- If pixel retrieval also fails, the deterministic predictor design is the more likely bottleneck.

The pixel-control protocol must be frozen before implementation. The completed latent formal run must not be rerun, retuned, or replaced. Gate 3 target-guided planning remains blocked.
