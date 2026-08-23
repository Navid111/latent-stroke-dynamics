# Controlled latent-planner result — 2026-08-23

**Status:** Frozen controlled criterion failed; implementation integrity passed  
**Rerun allowed:** No  
**Retuning on these targets:** No

## Integrity

All 89 tests passed before the single authorized run. All six target pairs completed. Checkpoint hashes matched, every learned trajectory replayed deterministically, each latent target was encoded once, exact observed canvases were re-encoded every step, predicted latents were never rolled forward, every model remained frozen, and no training or fine-tuning occurred.

## Aggregate result

| Method | Mean final MSE | Mean best MSE | Mean best step | Improvement from blank | Exact top-1 | Exact top-5 | Mean exact rank | Score–exact Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.146905 | 0.130692 | 48.0 | 9.79% | — | — | — | — |
| Exact pixel | **0.045569** | **0.045566** | 99.8 | **72.14%** | — | — | — | — |
| Learned pixel | 0.053630 | 0.051523 | 71.3 | 67.11% | 34.0% | 59.8% | 5.94 | — |
| Latent MSE | 0.076142 | 0.071020 | 65.0 | 52.57% | 9.5% | 27.2% | 21.53 | 0.549 |
| Latent ranking | 0.090934 | 0.083470 | 53.5 | 44.24% | 5.0% | 13.0% | 35.01 | 0.426 |

## Frozen decision

The ranking-aware latent planner:

- improved every target from blank: **pass**;
- reduced mean final MSE by 38.10% versus random, above the frozen 20% minimum: **pass**;
- reached 1.996× the exact-pixel error, above the frozen maximum of 1.5×: **fail**;
- passed implementation integrity: **pass**.

Because every required criterion had to pass, the controlled decision is **fail**.

## Interpretation

This is a scientific failure of the strong planner criterion, not a software failure and not evidence of zero usefulness. Both latent planners substantially improved the canvases relative to random, and latent ranking improved all six targets. However, the ranking-aware planner remained too far from the exact oracle and was weaker than both latent MSE and learned pixel.

Ranking-aware final error was 1.194× latent-MSE error and 1.696× learned-pixel error. Its selected stroke was the exact best candidate only 5% of the time, entered the exact top five only 13% of the time, and averaged rank 35.01 among 128 candidates. Its score-to-exact Spearman correlation was positive but lower than latent MSE's (0.426 versus 0.549).

The formal 74.44% result therefore remains valid but answers a narrower question: ranking loss improved four-way one-step action retrieval. That success did not transfer into superiority under repeated 128-candidate target-distance planning. MSE-style next-latent calibration appears more useful for the planner's full-grid target-distance score than the tested counterfactual-ranking objective.

The mean best step occurred earlier than the final step for every non-oracle method. Latent ranking was best around step 53.5 on average and ended about 8.9% worse than its preserved best frame, supporting the observed long-horizon overpainting limitation.

## Qualitative review

The montage agrees with the metrics. Exact-pixel and learned-pixel outputs preserve the target stroke layout most clearly. Latent-MSE outputs recover substantial coarse structure but add dense gray regions. Latent-ranking outputs show some target-aligned strokes but lose more fine structure and accumulate larger block-like gray regions. Random outputs are visibly cluttered.

## Thesis claim

A defensible conclusion is:

> Action-conditioned latent prediction can guide sequential stroke selection better than random in this controlled renderer, but one-step ranking-aware retrieval gains did not yield a competitive long-horizon planner under the tested target-distance scoring rule. Pixel-space prediction remained substantially stronger, and MSE-only latent dynamics outperformed ranking-aware latent dynamics in multi-step planning.

## Closure and next work

The controlled comparison is immutable and closed. Next work may analyze its saved per-target and per-step artifacts without generating new controlled data. Any further experiment must use new seeds, a separately frozen protocol, and a clearly exploratory or confirmatory label. The immediate thesis priority is to write the methodology, results, and discussion around the now-complete evidence chain.
