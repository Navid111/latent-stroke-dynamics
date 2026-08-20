# Current State

**Last updated:** 2026-08-20  
**Branch:** `main`  
**Current gate:** Gate 2 — deterministic one-step latent prediction  
**Gate status:** Development v2 reviewed; retrieval decomposition next

## Objective

Train the smallest action-conditioned model that predicts how one deterministic stroke changes frozen spatial canvas features, and determine whether it outperforms trivial baselines on held-out transitions.

## Completed

- Froze and passed Gate 1; archived its formal result.
- Read the focused literature-report scope and reconciled it with the project design.
- Froze the Gate 2 protocol before predictor implementation.
- Implemented deterministic transitions, split fingerprints, action encodings, identity/mean/linear/MLP predictors, balanced loss, spatial metrics, counterfactual retrieval, tests, caching, training, and plots.
- Passed the initial 14-test suite and completed engineering smoke 1.
- Found and repaired duplicate counterfactual outcomes; amended formal data seeds before any formal run because their old prefixes were exposed.
- Passed the revised 15-test suite in 1.99 seconds under Python 3.14.4.
- Completed development v2 with 256/64/96 examples and one model seed on the local M1.
- Verified unique rendered and encoded candidates, finite metrics, successful overfit check, and a stable end-to-end pipeline.
- Observed strong held-out average-error prediction and positive improvement at every crowding level.
- Observed counterfactual retrieval near chance and far below the frozen threshold.
- Added a no-retraining retrieval-decomposition script and unit tests.

## Gate 2 development-v2 result

Validation selected the linear predictor.

- action-region MSE: 0.000973;
- improvement versus identity: 57.2%;
- improvement versus mean delta: 51.4%;
- crowding improvements: +77.3%, +35.2%, and +9.3% for 0, 5, and 15 prior strokes;
- counterfactual retrieval: 22/96 = 22.9%;
- implementation sanity: passed;
- formal eligibility: false.

The predictor captures broad latent consequences but appears to smooth or underestimate the exact stroke effect. See `docs/gate-2-dev-v2.md`.

## Frozen Gate 2 decisions

- Frozen encoder: `facebook/dinov2-small`.
- Target: spatial patch-token residual `delta = z_next - z_current`.
- Deterministic one-step transitions only.
- Identity, mean-delta, linear, and small nonlinear predictors remain visible.
- Formal model seeds: `11`, `22`, and `33`.
- Untouched amended formal data seeds: `20260824`–`20260827`.
- Four rendered and encoded counterfactual outcomes must be distinct.
- Primary evidence: action-region error and counterfactual retrieval.
- No target-guided ranking, reinforcement learning, stochastic dynamics, or multi-step rollout.

## Validation status

Development v2 validates the implementation and supports one-step latent predictability under average error. It does not support precise counterfactual discrimination. No formal split has been generated or viewed, and no Gate 2 decision has been made.

The next step analyzes existing development outputs only. It must not retrain, re-encode, alter thresholds, or use `--formal-run`.

## Next actions

1. Pull and test the retrieval-diagnostic code.
2. Run `experiments/02b_retrieval_diagnostics.py` on `outputs/gate2-dev-v2`.
3. Inspect candidate preferences, pairwise win rates, margins, and metadata slices.
4. Decide whether the implementation is ready to freeze as-is for the formal run or whether a clearly identified implementation defect remains.
5. Freeze the exact formal training command before touching amended formal data.

## Immediate next step

Run the lightweight retrieval decomposition. Do not rerun DINOv2 encoding or predictor training.

## Handoff note

A mixed result is scientifically useful: low average latent error and weak exact action discrimination answer different questions. Preserve both rather than optimizing away the retrieval failure.
