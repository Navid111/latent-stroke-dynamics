# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Stage A planner-score development audit  
**Status:** One development score audit authorized; not yet executed

## Closed evidence

The single six-target, five-method controlled comparison remains closed. Mean final MSE was 0.146905 random, 0.045569 exact pixel, 0.053630 learned pixel, 0.076142 latent MSE, and 0.090934 latent ranking. Latent ranking improved every target and beat random by 38.10%, but its 1.996× exact-pixel ratio failed the frozen 1.5× maximum. Integrity passed completely.

## Read-only diagnosis

Forced continuation caused overpainting, but it was not sufficient to explain failure. Ranking's mean best MSE was 0.083470, about 1.832× exact pixel even with oracle best-frame selection. Ranking also had weaker top-1, top-5, mean-rank, regret, and Spearman diagnostics than latent MSE. The main follow-up hypothesis is planner-score misalignment; stopping is secondary.

## Stage A validation

The protocol was committed before implementation or data. The guarded runner then passed all 96 tests. Validation reported `planner_score_audit_runner_valid_unauthorized`, all closed resource references matched, all downstream phases were unauthorized, and no model, target, trajectory, candidate set, output, training, or fine-tuning was created.

## Current authorization

Exactly one development score audit is now authorized. It uses eight new target seeds, eight new state-planner seeds, eight new candidate seeds, 72 fixed candidate sets, 128 candidates per set, two frozen predictor ensembles, and five frozen scores. No training or fine-tuning is authorized. Planner development and confirmatory evaluation remain unauthorized.

## Next action

Pull the authorization commit, rerun the 96-test suite, and execute `python experiments/18_planner_score_alignment.py --development-score-audit` exactly once. Preserve any `.incomplete` directory if an error or interruption occurs.
