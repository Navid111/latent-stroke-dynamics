# Current State

**Last updated:** 2026-08-23  
**Branch:** `main`  
**Current stage:** Stage A planner-score audit validation  
**Status:** Guarded runner implemented; every new-data phase unauthorized

## Closed evidence

The single six-target, five-method controlled comparison remains closed. Mean final MSE was 0.146905 random, 0.045569 exact pixel, 0.053630 learned pixel, 0.076142 latent MSE, and 0.090934 latent ranking. Latent ranking improved every target and beat random by 38.10%, but its 1.996× exact-pixel ratio failed the frozen 1.5× maximum. Integrity passed completely.

## Read-only diagnosis

Forced continuation caused overpainting, but it was not sufficient to explain failure. Ranking's mean best MSE was 0.083470, about 1.832× exact pixel even with oracle best-frame selection. Ranking also had weaker top-1, top-5, mean-rank, regret, and Spearman diagnostics than latent MSE. The main follow-up hypothesis is planner-score misalignment; stopping is secondary.

## Stage A implementation

The protocol was committed before implementation or data. The guarded development runner now supports five exactly frozen scores across the MSE-only and ranking-aware three-seed ensembles. It includes inverse-standardized frozen decoding, pixel-error patch weights, Sobel scoring, exact candidate labels, deterministic state-bank generation, atomic outputs, overwrite refusal, frozen lexicographic selection, and complete integrity records.

Validation-only mode must remain side-effect free. Development, planner-development, and confirmatory authorizations are all false. The closed targets are prohibited, and no Stage A model training or fine-tuning is allowed.

## Next action

Pull the implementation, run the full test suite, and run only the validation command in `docs/planner-score-audit-command.md`. Do not run the development audit until its output is reviewed and a separate one-time authorization is committed.
