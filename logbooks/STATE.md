# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Stage A planner-score alignment protocol  
**Status:** Protocol frozen before implementation or new data

## Closed evidence

The single six-target, five-method controlled comparison remains closed. Mean final MSE was 0.146905 random, 0.045569 exact pixel, 0.053630 learned pixel, 0.076142 latent MSE, and 0.090934 latent ranking. Latent ranking improved every target and beat random by 38.10%, but its 1.996× exact-pixel ratio failed the frozen 1.5× maximum. Integrity passed completely.

## Read-only diagnosis

Forced continuation caused overpainting, but it was not sufficient to explain failure. Ranking's mean best MSE was 0.083470, about 1.832× exact pixel even with oracle best-frame selection. Ranking also had weaker top-1, top-5, mean-rank, regret, and Spearman diagnostics than latent MSE. The main follow-up hypothesis is planner-score misalignment; stopping is secondary.

## New frozen extension

`configs/planner-score-alignment-2026-08-23.json` and `docs/planner-score-alignment-protocol.md` freeze a post-controlled Stage A study before implementation or data. It compares five reference-target scores across both existing three-seed latent ensembles on 72 new development state/candidate sets. A later planner development phase tests the selected score with and without an untuned no-op rule. Six additional targets are reserved for one separately authorized confirmatory comparison.

No training or fine-tuning is allowed in Stage A. All new seeds are disjoint, and closed targets are prohibited.

## Next action

Implement the guarded validation-only score-audit runner and tests. Validation must not load models, generate targets or state trajectories, create candidate sets, train models, or create output directories.
