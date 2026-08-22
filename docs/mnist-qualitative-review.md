# MNIST-3 qualitative painter review — 2026-08-22

**Role:** Post-controlled qualitative diagnostic  
**Target:** MNIST-style white 3 on black, automatically normalized to dark-on-white  
**Budget:** 100 strokes, 128 candidates, seed `20261001`  
**Controlled Stage 3 decision:** Unchanged

## Polarity failure and repair

The first 20-step run preserved the input's light-on-dark polarity. Because the renderer starts white and adds only dark strokes, it optimized the black background rather than the white digit. The visually invalid output still reduced MSE by 33.23%, demonstrating that a background-dominated pixel metric can improve while structure remains wrong.

Automatic border-based polarity normalization converted the target to the renderer-compatible dark-on-white convention. The repaired 20-step learned smoke then improved MSE by 64.22%, achieved 85% exact top-1 and 100% top-5 agreement among 32 candidates, and produced a recognizable angular approximation.

## Full-budget comparison

| Metric | Learned | Exact greedy |
|---|---:|---:|
| Initial MSE | 0.144356 | 0.144356 |
| Final MSE | 0.040860 | **0.022196** |
| Final MAE | 0.099737 | **0.060518** |
| Relative MSE improvement | 71.70% | **84.62%** |
| Improving steps | 42/100 | **88/100** |
| Best step from saved curve | 33 | 99 |
| Runtime with diagnostics | 5.15 s | 1.36 s |

The learned final MSE was approximately `1.841` times the exact final MSE. Exact greedy was visibly cleaner and more recognizably shaped as a 3.

## Learned ranking diagnostics

- exact top-1 agreement: 23%;
- exact top-5 agreement: 37%;
- mean exact rank: 9.23 of 128;
- mean one-step regret: `0.000410`;
- maximum regret: `0.001125`.

The learned curve reached its minimum at step 33 and then gradually deteriorated. The exact curve continued improving almost to the full budget, reaching its minimum at step 99. Therefore, forced strokes/no no-op are a secondary issue on this target; the principal late-stage limitation is learned candidate ranking.

## Interpretation

The most plausible mechanism is state-distribution shift. The checkpoint was trained on one-step transitions whose base canvases contained 0, 5, or 15 random prior strokes. A 100-step target-guided trajectory creates denser, structured overlap patterns well outside that training distribution. The model observes the exact canvas after every stroke, so there is no latent-state drift, but small transition-scoring errors cause it to choose increasingly suboptimal actions in crowded states.

Increasing the candidate set from 32 to 128 also makes exact top-1 agreement harder and introduces more near-tied alternatives. Nevertheless, exact greedy shows that strongly improving candidates remained available long after the learned curve turned upward.

## Artifact consequence

The reliable arbitrary-image renderer is currently exact greedy. The learned painter remains scientifically useful as:

1. a successful controlled synthetic planner;
2. a demonstration that low one-step regret can support short trajectories;
3. a documented long-horizon, out-of-distribution limitation.

The user-facing command now saves both the literal requested-budget final canvas and the lowest-MSE `best_painting.png`, together with `best_step`, `best_mse`, and final-versus-best metrics. This does not alter any trajectory or controlled result.

## Thesis consequence

Do not claim robust arbitrary-image learned painting. The defensible conclusion is that the learned pixel dynamics model supports planning on the frozen controlled synthetic distribution but degrades during long, crowded qualitative trajectories. Exact greedy remains a strong artifact baseline and exposes the gap between transition learning and robust sequential deployment.
