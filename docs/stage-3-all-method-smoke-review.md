# Stage 3 all-method smoke review — 2026-08-22

**Status:** Engineering pass  
**Role:** Development-only learned-planning check  
**Controlled Stage 3 decision:** Not yet made

## Primary result

| Method | Final MSE | Improvement from white | Final MAE | Improving steps |
|---|---:|---:|---:|---:|
| Random | 0.144913 | 3.03% | 0.221286 | 11/20 |
| Exact greedy | 0.074427 | 50.20% | 0.134931 | 20/20 |
| Learned | 0.089196 | 40.32% | 0.159811 | 19/20 |

The learned planner’s final MSE was approximately 38.45% below random and 19.84% above exact greedy. The latter corresponds to a learned/exact ratio of `1.198437`, inside the preregistered controlled reference of `1.25`. This smoke cannot itself satisfy that controlled criterion because it used one development target and smaller budgets.

## Learned ranking diagnostics

- exact top-1 agreement: 55%;
- exact top-5 agreement: 80%;
- mean exact rank: 4.6 of 32;
- mean one-step exact regret: `0.000653`;
- maximum regret: `0.003688` at step 7;
- deterministic learned replay: passed.

The model does not need exact top-1 agreement at every step to plan usefully. Most non-top-1 choices have small regret and still improve the true canvas. One action at step 7 worsened MSE slightly; subsequent replanning recovered and the final trajectory remained much stronger than random.

## Visual review

The learned result is visibly intermediate between random and exact greedy. It captures the dominant dark diagonal and several broad target structures but misses or displaces more thin details than exact greedy. The fixed-scale error panel is correspondingly lower than random and higher than exact.

The progress plot shows learned planning tracking exact closely for roughly the first five actions. It then diverges after several ranking errors but continues a strong downward trend. Random remains near the initial error.

## Runtime

The learned trajectory took `0.429` seconds for 20 steps and 32 candidates per step, versus `0.112` seconds for exact and `0.091` seconds for random on the M1. Learned inference is slower but still comfortably practical for the planned six-target controlled run.

## Decision

No engineering repair or retuning is justified. The checkpoint, candidate scoring, exact execution, replanning, diagnostics, figures, and deterministic replay behave coherently. The six-target, 100-step, 128-candidate controlled configuration is now frozen and must not be changed after its results are visible.
