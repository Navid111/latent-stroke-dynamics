# Stage 3 controlled result — 2026-08-22

**Status:** Success  
**Controlled eligible:** Yes  
**Implementation integrity:** Passed  
**Protocol:** Frozen before data generation  
**Rerun or retuning allowed:** No

## Decision

The learned pixel planner passed every frozen Stage 3 criterion on the single authorized six-target run.

| Frozen criterion | Required | Observed | Decision |
|---|---:|---:|---|
| Learned improves every target from white | 6/6 | 6/6 | Pass |
| Mean learned final-MSE reduction versus random | ≥20% | 61.51% | Pass |
| Mean learned/exact final-MSE ratio | ≤1.25 | 1.1893 | Pass |
| Implementation integrity | Pass | Pass | Pass |

## Aggregate results

| Method | Mean initial MSE | Mean final MSE | Mean final MAE | Mean improvement | Improving steps | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.184693 | 0.155971 | 0.308397 | 14.60% | 54.17/100 | 1.74 s |
| Exact greedy | 0.184693 | 0.050479 | 0.109828 | 72.27% | 97.17/100 | 2.22 s |
| Learned | 0.184693 | 0.060032 | 0.125284 | 66.69% | 69.67/100 | 8.08 s |

The learned planner finished 61.51% below random in final MSE and 18.93% above exact greedy. The absolute learned/exact gap was `0.009553` mean final MSE.

## Learned ranking diagnostics

- exact top-1 agreement: 33.5%;
- exact top-5 agreement: 58.67%;
- mean exact rank: 7.51 of 128;
- mean exact one-step regret: `0.0003867`;
- maximum exact one-step regret: `0.007298`;
- deterministic replay: passed for every target.

Top-1 agreement is lower than in the 32-candidate development smoke, which is expected under a four-times-larger candidate set and a five-times-longer trajectory. More importantly, low regret let many non-top-1 selections remain useful. Exact agreement is therefore not required at every action for successful target-guided painting.

## Per-target learned result

| Target | Initial MSE | Learned final MSE | Learned improvement | Exact final MSE | Top-1 | Top-5 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.149447 | 0.070200 | 53.03% | 0.047307 | 20% | 29% | 14.59 |
| 2 | 0.167305 | 0.054448 | 67.46% | 0.048940 | 34% | 64% | 5.07 |
| 3 | 0.241001 | 0.061314 | 74.56% | 0.054569 | 32% | 66% | 6.27 |
| 4 | 0.191256 | 0.065325 | 65.84% | 0.053098 | 34% | 61% | 5.41 |
| 5 | 0.181361 | 0.045376 | 74.98% | 0.053530 | 54% | 92% | 2.75 |
| 6 | 0.177787 | 0.063530 | 64.27% | 0.045431 | 27% | 40% | 10.98 |

Targets 1 and 6 were the most difficult ranking cases. Target 5 is scientifically useful: learned planning finished better than exact greedy. This is not an implementation contradiction because “exact” is a one-step greedy oracle, not a globally optimal 100-step planner. Different early selections create different future candidate distributions, so a non-greedy trajectory can occasionally reach a better final canvas.

## Curve and montage review

The aggregate curve shows exact and learned planning reducing error sharply during the first 20–30 strokes. Exact continues improving gradually to the end. Learned reaches roughly `0.060` by 60–70 strokes and then plateaus with a slight late increase, while random stalls and eventually worsens. This supports both the success claim and a practical future improvement: permit early stopping or a no-op action after progress saturates.

The montage is consistent with the metrics. Random outputs are heavily cluttered and fail to preserve the target’s main organization. Exact greedy generally reconstructs the dominant geometry most cleanly. Learned outputs preserve the principal dark and gray structures across all six targets and are visibly much closer to exact than random, while thin lines and crowded intersections remain the main failure modes.

## What this establishes

A tiny learned full-resolution pixel dynamics model can support sequential target-guided stroke selection. Under the frozen controlled protocol, it substantially outperforms random candidate selection and approaches an exact one-step renderer oracle.

## What this does not establish

- The controlled targets are synthetic and built from the same stroke family used by the renderer.
- The output is 64×64 grayscale with straight lines only.
- The proposal distribution is target-aware; the comparison isolates candidate ranking under the same proposal mechanism.
- Exact greedy is a one-step oracle, not a global planner.
- This does not convert latent Gate 2 into a pass or prove DINOv2 alone caused the latent failure.
- Arbitrary natural-image performance remains qualitative and must be tested separately.

## Next stage

Package the frozen learned planner as a user-facing image-to-strokes command, run a small set of arbitrary-image demonstrations, preserve representative successes and failures, and then integrate the controlled result into the thesis Methods, Results, and Discussion chapters.
