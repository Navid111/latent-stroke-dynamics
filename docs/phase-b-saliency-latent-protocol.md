# Frozen Phase B0 protocol — saliency-guided latent painter foundation

**Frozen:** 2026-08-23, before implementation or Phase B data  
**Branch:** `phase-b/saliency-latent`  
**Base commit:** `c211c3ab3a37b9c37eda5ba3c07c01173fd4c7f7`  
**Status:** implementation and all experimental data unauthorized

## Purpose

Phase B studies whether a stronger action-conditioned joint-embedding model can turn latent prediction into better long-horizon stroke decisions. It does not reopen Stage A, alter any archived result, or reuse any closed target.

The existing system already predicts the next spatial task-autoencoder latent from the current latent and a proposed stroke. Formal four-way retrieval reached 74.44% with ranking-aware training, but the ranking-aware model was weaker in 128-candidate long-horizon planning. A score-only change to normalized-latent L1 modestly improved the frozen MSE-only planner, while a direct exact-current versus predicted-candidate no-op comparison stopped prematurely.

Phase B0 addresses the remaining model question directly: learn a spatial joint embedding and action-conditioned dynamics predictor together, then supervise a separate progress head on exact target-error reduction so stroke candidates and no-op are evaluated on one calibrated learned scale.

## Scope separation

Phase B is deliberately separated into three scientific units.

1. **B0 — latent foundation:** 64×64 grayscale, unchanged straight-stroke renderer, new joint-embedding dynamics and calibrated progress model.
2. **B1 — region scheduling:** background→object→detail allocation using saliency, detections, edges, and residual error. B1 remains unauthorized and requires its own frozen protocol after B0 closes successfully.
3. **B2 — RGB/high-resolution application:** color, alpha, richer strokes, and high-resolution replay. B2 remains unauthorized and cannot change B0 evidence.

This prevents color, resolution, renderer, saliency, representation, and objective changes from becoming scientifically inseparable.

## Research questions

### Primary

Does the frozen Phase B0 planner-aligned joint-embedding model reduce mean long-horizon final MSE by at least 5% relative to the archived MSE-only ensemble with normalized-latent L1 on new development targets, while improving every target from blank and reaching at most 1.5× exact-pixel error?

### Secondary

1. Does a trainable multi-scale joint embedding retain action-discriminative one-stroke information without reconstruction?
2. Does rendered spatial action conditioning improve candidate consequence prediction over a raw action vector?
3. Does exact-progress supervision reduce 128-way candidate regret?
4. Does an explicit no-op passed through the same learned pipeline avoid the Stage A calibration failure?

## Immutable historical boundaries

- Gate 1, Gate 2, pixel control, Stage 3, representation extension, ranking development, formal ranking, planner smoke, controlled planner, score audit, and planner development remain closed.
- No closed target, state trajectory, candidate set, output, or reserved confirmatory seed may be reused.
- The Stage A `not_eligible` decision remains unchanged.
- The archived task autoencoder, latent statistics, six formal latent predictors, learned-pixel checkpoint, and their hashes remain immutable comparators.
- No Phase B result may retroactively change a historical pass/fail decision.
- The system must be described as action-conditioned and joint-embedding or JEPA-inspired, not as a canonical JEPA implementation.

## Fixed renderer and transition distribution

B0 retains the exact 64×64 grayscale straight-stroke renderer.

- normalized endpoints in `[0, 1]`;
- widths `1, 2, 3, 4`;
- grayscale values `0, 32, 64, 96, 128`;
- minimum sampled length `0.20` for transition training;
- transition crowding `0, 5, 15, 30`;
- exact renderer execution for every planner action;
- target compositions contain 20 strokes;
- no opacity, color, curves, anti-aliasing, or renderer changes in B0.

Ten percent of transition examples are explicit no-op transitions. Their action raster is zero and their exact next canvas equals the current canvas bit-for-bit.

## Model architecture

### Online multi-scale encoder

The online encoder receives `[N, 1, 64, 64]` canvases. Every convolution includes a bias. Hidden convolutions use GroupNorm followed by GELU.

| Stage | Operation | Output |
| --- | --- | --- |
| Stem | Conv 1→24, 3×3, stride 1, padding 1; GN(6); GELU | 24×64×64 |
| 64-stage | Conv 24→24, 3×3, stride 1, padding 1; GN(6); GELU | 24×64×64 |
| Down-32 | Conv 24→48, 4×4, stride 2, padding 1; GN(8); GELU | 48×32×32 |
| 32-stage | Conv 48→48, 3×3, stride 1, padding 1; GN(8); GELU | 48×32×32 |
| Down-16 | Conv 48→64, 4×4, stride 2, padding 1; GN(8); GELU | 64×16×16 |
| 16-stage | Conv 64→64, 3×3, stride 1, padding 1; GN(8); GELU | 64×16×16 |
| Projection-32 | Conv 48→32, 1×1 | 32×32×32 |
| Projection-16 | Conv 64→64, 1×1 | 64×16×16 |

The two projected maps are the learned representation. No decoder participates in B0 training or evaluation.

### Target encoder

The target encoder is an exact architecture copy initialized from the online encoder. It receives no gradient and is updated after every optimizer step by exponential moving average with momentum `0.99`. Target features are stop-gradient.

### Action raster and encoder

Each non-no-op stroke becomes a two-channel 64×64 raster:

1. binary stroke coverage;
2. coverage multiplied by grayscale value divided by 255.

No-op is exactly zero in both channels.

The action encoder is:

| Stage | Operation | Output |
| --- | --- | --- |
| Action-32 | Conv 2→16, 5×5, stride 2, padding 2; GN(4); GELU | 16×32×32 |
| Action-16 | Conv 16→32, 3×3, stride 2, padding 1; GN(8); GELU | 32×16×16 |

### Multi-scale latent predictor

The predictor outputs a residual at each scale.

- 32-scale: concatenate current 32-scale latent and Action-32; Conv 48→64, 3×3, padding 1; GN(8); GELU; Conv 64→32, 3×3, padding 1.
- 16-scale: average-pool the predicted 32-scale residual to 16×16; concatenate current 16-scale latent, Action-16, and the pooled 32 channels; Conv 128→96, 3×3, padding 1; GN(8); GELU; Conv 96→64, 3×3, padding 1.
- Predicted next latent equals current latent plus predicted residual at each scale.

### Progress head

The progress head receives global-average-pooled 16-scale vectors for current latent, target latent, predicted-next latent minus target latent, and Action-16. The 224-dimensional input passes through Linear 224→128, GELU, Linear 128→64, GELU, Linear 64→1.

The scalar target is exact pixel-MSE reduction:

`MSE(current, target) - MSE(exact_next, target)`.

Targets are standardized with training-only mean and standard deviation. The saved statistics are immutable after training. No-op has exact progress zero.

The complete trainable model must contain no more than 500,000 parameters. Exact component and total counts must be recorded in a separate implementation manifest before any development authorization.

## Fixed objectives

### Joint prediction

At each scale, use balanced Smooth L1 between predicted and target next latents, weighting action-covered and uncovered spatial positions equally. Scale weights are 0.5 at 32×32 and 0.5 at 16×16.

### Anti-collapse

At both scales, compute variance and covariance regularization across batch and spatial positions:

- variance floor `1.0`;
- variance weight `0.10`;
- off-diagonal covariance weight `0.01`.

### No-op consistency

For explicit no-op examples, penalize predicted residual magnitude at both scales with weight `0.25`.

### Planner-aligned terms

For each planner-supervision state, evaluate exactly 32 candidates: one explicit no-op and 31 target-guided strokes. Exact rendering supplies progress labels.

- Progress regression: Smooth L1/Huber on standardized exact progress, weight `1.0`, beta `1.0`.
- Candidate ranking: cross-entropy over predicted standardized progress with the exact best candidate as label, weight `0.30`, temperature `0.10`.
- Exact ties use tolerance `1e-12` and then the lowest candidate index. No-op is always index zero.

Two development variants share the same architecture, initialization seed, transition data, and optimizer:

1. `joint_prediction_only` — joint prediction + anti-collapse + no-op consistency;
2. `joint_prediction_progress` — the same terms + progress regression + candidate ranking.

No loss-weight or architecture grid is allowed.

## Training settings

- Device: CPU.
- Optimizer: AdamW.
- Learning rate: `0.0003`.
- Weight decay: `0.0001`.
- Batch size: `16`.
- Maximum epochs: `40`.
- Early-stopping patience: `8`.
- Gradient clipping norm: `5.0`.
- Development model seed: `73` for both matched variants.
- Maximum completed development execution: one.
- Total Phase B0 development training wall-clock cap: six hours on the documented Apple Silicon machine.
- If the cap is reached, preserve a `compute_limit` result; do not silently enlarge the budget.

## Development data — unauthorized

All development seeds are new and disjoint from prior work.

### Transition splits

| Split | Samples | Seed |
| --- | ---: | ---: |
| Train | 2,048 | 20270401 |
| Validation | 512 | 20270402 |
| Diagnostic test | 512 | 20270403 |

Diagnostic test cannot select settings.

### Planner-supervision train bank

- target seeds `20270411`–`20270418`;
- state-trajectory seeds `20270421`–`20270428`;
- candidate seeds `20270431`–`20270438`;
- eight targets and eight target-independent states per target;
- states: blank; exact steps 20, 40, 60; random steps 20, 40, 60, 80;
- 32 candidates per state including no-op.

### Planner-supervision validation bank

- target seeds `20270441`–`20270444`;
- state-trajectory seeds `20270451`–`20270454`;
- candidate seeds `20270461`–`20270464`;
- four targets with the same eight-state definition;
- 32 candidates per state including no-op.

### Long-horizon development

- target seeds `20270471`–`20270473`;
- planner seeds `20270481`–`20270483`;
- 100 maximum steps;
- 128 candidates per decision, including no-op only for the calibrated-progress method;
- exact execution and observed-canvas re-encoding after every chosen action.

Methods:

1. exact pixel;
2. learned pixel;
3. archived MSE-only ensemble with normalized-latent L1, forced horizon;
4. new `joint_prediction_only`, forced horizon;
5. new `joint_prediction_progress`, forced horizon;
6. new `joint_prediction_progress` with explicit no-op.

## Development metrics

Report, without selective omission:

- joint prediction loss by scale;
- latent feature mean standard deviation and covariance summary;
- four-way exact-action retrieval;
- 32-way and 128-way top-1/top-5, mean rank, regret, and Spearman;
- progress MAE, RMSE, sign accuracy, and calibration by predicted-progress bin;
- premature-stop rate when an improving exact candidate exists;
- final and best pixel MSE/MAE, best step, executed steps, and stop rate;
- exact, learned-pixel, archived-latent, and new-model comparisons;
- wall-clock, parameter counts, seeds, hashes, deterministic replay, and integrity checks.

## Frozen development eligibility

Formal B0 evaluation is eligible only if every condition passes:

1. implementation integrity and deterministic replay pass;
2. no historical or closed artifact changes;
3. representation does not collapse: mean channel standard deviation is at least 0.50 at both scales on diagnostic data;
4. diagnostic four-way retrieval for `joint_prediction_progress` is at least 50%;
5. mean 128-way exact regret is at least 10% lower than the archived MSE-only + normalized-latent L1 comparator;
6. the no-op method improves every long-horizon development target from blank;
7. its mean final MSE is at least 5% lower than the archived comparator;
8. its mean final MSE is no worse than the new prediction-only forced planner;
9. its mean final MSE is at most 1.5× exact-pixel mean final MSE;
10. premature-stop rate is at most 10% on evaluated states with at least one exactly improving stroke;
11. the six-hour compute cap is respected.

Failure of any condition closes B0 as `not_eligible`. Criteria cannot be weakened after results are visible.

## Reserved formal B0 — unauthorized

Formal data may be generated once only after a separate authorization following successful development closure.

### Transition splits

| Split | Samples | Seed |
| --- | ---: | ---: |
| Train | 4,096 | 20270501 |
| Validation | 1,024 | 20270502 |
| Test | 1,024 | 20270503 |
| Crowding-60 stress | 256 | 20270504 |

### Planner supervision

- train targets `20270511`–`20270518`, trajectories `20270521`–`20270528`, candidates `20270531`–`20270538`;
- validation targets `20270541`–`20270544`, trajectories `20270551`–`20270554`, candidates `20270561`–`20270564`.

### Long-horizon formal targets

- target seeds `20270571`–`20270576`;
- planner seeds `20270581`–`20270586`;
- model seeds `83, 97, 109`;
- 100 maximum steps and 128 candidates per decision.

Formal success requires the same integrity, improvement-from-blank, 5% mean reduction, 1.5× exact-pixel ratio, and 10% premature-stop maximum across the six formal targets. Formal data cannot select or tune anything.

## Validation-only boundary

Before development authorization, code may:

- load and validate this JSON;
- instantiate randomly initialized Phase B models;
- use deterministic random dummy tensors;
- check shapes, gradients, EMA updates, no-op identity handling, objective finiteness, and parameter counts;
- run tiny in-memory overfit checks on dummy tensors;
- verify closed resource references and seed disjointness.

It may not:

- load historical checkpoints;
- generate renderer transitions, targets, state banks, or candidate sets;
- create output directories;
- train on real or synthetic renderer data;
- download a new model;
- authorize development, formal B0, B1, or B2.

## Next action

Implement only the configuration validator, architecture, objective utilities, and validation-only tests. Run the complete test suite. Development remains unauthorized until a separate commit records a passing validation result and explicitly authorizes one execution.