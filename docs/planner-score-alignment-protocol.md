# Frozen planner-score alignment protocol

**Frozen:** 2026-08-23, before implementation or new data  
**Role:** post-controlled exploratory extension  
**Closed result:** the six-target controlled failure remains immutable

## Motivation

The closed controlled study showed two separable problems. Latent ranking ended about 8.9% above its mean best frame, so forced continuation caused overpainting. However, oracle best-frame selection would still leave ranking at about 1.832× exact-pixel error, above the frozen 1.5× criterion. Candidate ordering was also weaker for ranking than for latent MSE: 5.0% versus 9.5% exact top-1, 13.0% versus 27.2% top-5, mean exact rank 35.01 versus 21.53, and score-to-exact Spearman 0.426 versus 0.549.

This extension therefore tests score alignment before any new training. It does not revise, rerun, or retune the closed result.

## Question

Can a different reference-target score extract better candidate ordering from either already-frozen latent predictor ensemble, and can an untuned no-op rule reduce forced overpainting?

## Boundaries

- The closed smoke and controlled targets are prohibited.
- The task autoencoder, latent statistics, six formal predictors, and learned-pixel comparator remain frozen at their committed hashes.
- No training or fine-tuning is allowed in Stage A.
- Every target, planner, and candidate seed below is new and disjoint from prior experiments.
- Development may select exactly one predictor-family/score pair. Reserved confirmatory data may be run once only after a separate authorization.
- A new result cannot change the archived controlled `fail` or the formal one-step result.

## Stage A1 — development score audit

Eight new targets use seeds `20270101`–`20270108`. Each target has a fixed state bank independent of the audited scores:

- the blank canvas;
- exact-pixel trajectory steps 20, 40, 60, and 80;
- random trajectory steps 20, 40, 60, and 80.

State trajectories use seeds `20270111`–`20270118`. Independent audit candidate seeds are `20270121`–`20270128`. There are nine states per target, 72 state/candidate sets in total, and 128 unique target-guided candidates per state.

Both three-seed frozen ensembles are evaluated:

1. `mse_only`;
2. `ranking_aware`.

For every predicted next latent, evaluate exactly five scores:

1. **normalized latent MSE** — the closed full-grid score;
2. **normalized latent L1** — full-grid absolute error after per-patch L2 normalization;
3. **pixel-error-weighted normalized latent MSE** — patch MSE weighted by pooled exact current-to-target absolute pixel error plus its patch mean, normalized to unit mean weight;
4. **decoded pixel L1** — inverse-standardize the predicted latent, decode it with the frozen autoencoder, and compare it with exact target pixels;
5. **decoded pixel L1 + 0.25 Sobel L1** — add quarter-weighted Sobel-x/y response error. The quarter factor compensates for the four-unit absolute kernel scale and is frozen, not tuned.

The exact label is the target pixel MSE after exact candidate rendering. No no-op candidate is included in this one-step audit.

### Reported metrics

- exact top-1 and top-5 rates;
- mean exact selected rank;
- mean and maximum exact regret;
- mean score-to-exact Spearman correlation;
- runtime and complete per-state records.

### Frozen selection order

Select one predictor-family/score pair using, in order:

1. lowest mean exact regret;
2. highest exact top-5 rate;
3. highest mean Spearman correlation;
4. fixed score simplicity order as listed above;
5. `mse_only` before `ranking_aware`.

This allows the current baseline to win. Development is not a confirmatory test.

## Stage A2 — planner development

Three further targets use target seeds `20270201`–`20270203` and planner seeds `20270211`–`20270213`. Compare:

1. exact pixel;
2. learned pixel;
3. current latent-MSE ensemble with the closed score and a forced 100-step horizon;
4. the Stage A1 winner with a forced 100-step horizon;
5. the Stage A1 winner with a no-op rule.

The no-op score is computed on the exactly observed current state using the same selected score. It stops when the current-state score is less than or equal to the minimum predicted candidate score. The margin is fixed at zero and cannot be tuned.

Confirmatory eligibility requires implementation integrity, exact reuse of the Stage A1 winner, improvement from blank on every development target, and non-negative mean improvement versus the current forced latent-MSE baseline.

## Reserved confirmatory comparison

Six untouched targets use seeds `20270301`–`20270306`; planner seeds are `20270311`–`20270316`. The five Stage A2 methods run once with a 100-step maximum and 128 candidates per decision.

The selected no-op method succeeds only if all conditions hold:

1. improves every target from blank;
2. reduces mean final MSE by at least 5% versus current forced latent MSE;
3. reaches no more than 1.5× exact-pixel mean final MSE;
4. passes every implementation-integrity check.

Outperforming learned pixel is reported but not required.

## Why these losses

Pixel L1 is directly aligned with the reference image and is used widely in neural painting. Sobel adds a lightweight structural signal appropriate for 64×64 grayscale strokes. Error weighting focuses the latent metric on unresolved target regions. OT/Sinkhorn and VGG/LPIPS are not included in this first audit: OT's main gradient advantage is less direct for discrete candidate selection, while generic perceptual networks were already poorly aligned with this small grayscale domain.

## Interpretation

- A new score winning with frozen models indicates a planner-objective mismatch rather than a need for immediate retraining.
- No-op improvement without score improvement isolates forced-horizon overpainting.
- Failure of all frozen-score variants motivates, but does not automatically authorize, a separate Stage B predictor trained with next-latent MSE plus planner-aligned target-progress ranking.
- Every outcome remains a post-controlled extension and must be reported alongside the archived negative result.
