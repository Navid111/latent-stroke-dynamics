# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Stage A long-horizon planner development  
**Status:** One planner-development comparison authorized; not yet executed

## Closed evidence

The controlled multi-step comparison remains closed with complete integrity and a frozen criterion failure: latent ranking mean final MSE was 1.996× exact pixel, above the 1.5× maximum.

The single development score audit also completed and is closed. It evaluated 10 predictor/score pairs over 72 candidate sets without training or closed-target reuse. Implementation integrity passed.

## Frozen Stage A selection

The selected development pair is the three-seed MSE-only ensemble with normalized-latent L1 score. Mean exact regret was 0.001052, top-5 rate was 51.39%, mean exact rank was 12.39, and mean score-to-exact Spearman was 0.619. Compared with the previous normalized-latent MSE score using the same predictors, L1 reduced mean regret by about 28.8% and improved top-5 rate by 18.06 percentage points.

## Validation and authorization

The guarded long-horizon runner passed all 103 tests. Validation returned `planner_score_planner_development_runner_valid_unauthorized`; the archived selection, pixel checkpoint, autoencoder, statistics, and all six latent predictors were verified. No model was loaded, no target or planner data was generated, and no training or fine-tuning occurred.

Exactly one three-target planner-development comparison is now authorized. The selected L1 score, zero no-op margin, five methods, 100-step maximum, 128 candidates, seeds, and eligibility criteria remain frozen. Confirmatory evaluation is unauthorized.

## Next action

Pull the authorization commit, rerun the 103-test suite, and execute `python experiments/19_planner_score_development.py --planner-development` exactly once. Preserve any `.incomplete` directory after an error or interruption.
