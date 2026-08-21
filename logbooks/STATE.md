# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current gate:** Gate 2 complete — formal fail  
**Gate status:** Closed and fully diagnosed; pixel-space control next

## Formal Gate 2 decision

The exact frozen run completed with `formal_eligible: true` and selected the MLP family by validation error.

- action-region MSE: 0.000860;
- improvement versus identity: 61.8%;
- improvement versus mean delta: 57.1%;
- crowding improvements: +79.0%, +43.3%, and +25.0%;
- counterfactual retrieval: 27.7%;
- all seeds beat identity;
- overfit, finite-metric, and candidate-uniqueness checks passed.

The gate **failed** because retrieval was below the frozen 50% requirement and 35% fail boundary.

## Final retrieval diagnosis

The selected MLP was stable across seeds:

- seed accuracies: 26.3%, 27.3%, and 29.3%;
- seed standard deviation: 1.53 percentage points;
- width-changed candidate selected: 48.2%;
- true beats shifted position: 77.9%;
- true beats changed intensity: 75.2%;
- true beats changed width: 40.7%.

The model is not broadly action-insensitive. Its dominant failure is precise width discrimination. More data improved average prediction and other action dimensions but left width pairwise accuracy essentially unchanged from development.

## Completed

- Gate 1 formally passed and was archived.
- Gate 2 protocol, amendment, implementation, development diagnostics, and formal command were committed before formal evaluation.
- The formal run used untouched amended data and all three preregistered model seeds.
- Formal tables and plots were reviewed and agreed.
- Multi-seed retrieval diagnostics passed 19 local tests and were archived.
- Gate 2 is closed without rerunning or post-test tuning.

## Next actions

1. Freeze the minimal action-conditioned pixel-space control before implementation.
2. Use it to distinguish a latent-target/objective bottleneck from a general deterministic-predictor bottleneck.
3. Preserve identity, mean-delta, linear, and small nonlinear comparisons.
4. Do not begin Gate 3 target-guided planning with the failed latent predictor.
5. Treat contrastive, spatially interacting, or width-specific latent objectives as future work or explicitly post-formal ablations.

## Immediate next step

Write and freeze the pixel-space control protocol, then implement its smallest deterministic version. No further latent Gate 2 runs are authorized.
