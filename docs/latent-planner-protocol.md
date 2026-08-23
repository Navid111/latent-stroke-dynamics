# Frozen latent-space planner protocol

**Frozen:** 2026-08-23, before implementation and planner data  
**Role:** post-formal deployment study  
**Historical result:** the ranking-aware formal comparison remains closed and unchanged

## Question

Can the already-trained action-conditioned latent predictors support sequential target-guided stroke selection when used in an observe–predict–execute–re-encode loop?

The formal experiment established one-step action discrimination. This study tests whether that information transfers to multi-step planning. It is a separate experiment and cannot revise the formal claim.

## Fixed planner mechanism

At every step:

1. generate the same deterministic target-guided candidate strokes used by Stage 3;
2. encode the exactly observed current canvas with the frozen task autoencoder;
3. encode the target once with the same autoencoder and frozen channel statistics;
4. for each candidate, give the current latent, action vector, and action mask to each frozen predictor in the relevant three-seed ensemble;
5. form each predicted next latent as current latent plus predicted residual;
6. L2-normalize every patch feature and calculate full-grid MSE to the normalized target latent;
7. average candidate scores across seeds 11, 22, and 33;
8. execute the minimum-score stroke with the exact renderer;
9. re-encode the observed exact canvas before the next decision.

The planner never decodes a predicted latent and never rolls a predicted latent forward as state. Proposal generation remains pixel-error-guided and target-value-matched, as in Stage 3; only candidate scoring is latent.

## Frozen models

No model may be trained or fine-tuned.

- frozen task autoencoder and latent statistics from the completed representation extension;
- formal MSE-only latent checkpoints for seeds 11, 22, and 33;
- formal ranking-aware latent checkpoints for seeds 11, 22, and 33;
- frozen Stage 3 learned-pixel checkpoint for comparison.

Checkpoint file hashes must be measured in validation-only mode and committed before any planner smoke or controlled target is generated.

## Methods

1. `random` — random selection from the shared proposal process;
2. `exact_pixel` — exact rendered candidate with lowest target pixel MSE;
3. `learned_pixel` — frozen Stage 3 pixel predictor;
4. `latent_mse` — mean candidate score across the three formal MSE-only latent predictors;
5. `latent_ranking` — mean candidate score across the three formal ranking-aware latent predictors.

No single latent seed is selected using formal test or stress performance. The three-seed mean is fixed before planner data.

## Shared renderer and proposals

- canvas: 64×64 grayscale;
- target: 20 synthetic straight strokes;
- executed steps: 100;
- candidates per step: 128;
- error-guided candidate fraction: 0.80;
- candidate length range: 0.10–0.60;
- widths: 1, 2, 3, 4;
- values: 0, 32, 64, 96, 128;
- exact execution after every selection;
- best frame and requested final frame both preserved.

## Phases and untouched seeds

### Foundation validation

May load existing checkpoints and run in-memory synthetic shape/determinism checks. It must not generate smoke or controlled planner targets and must not train any model.

### Implementation smoke — unauthorized until hash freeze

- one synthetic target;
- target seed `20261201`;
- planner seed `20261202`;
- 20 steps;
- 32 candidates;
- all five methods;
- implementation diagnostics only;
- cannot select settings or revise the frozen protocol.

### Single controlled comparison — unauthorized until smoke review

Six new synthetic targets:

- target seeds `20261211`–`20261216`;
- planner seeds `20261221`–`20261226`;
- 100 steps and 128 candidates;
- all five methods;
- one execution only;
- atomic incomplete output and overwrite refusal.

## Primary deployment outcome

Mean final target pixel MSE across the six targets.

The `latent_ranking` planner is considered deployment-usable in this controlled study only if all conditions hold:

1. final MSE improves from the blank initial canvas on every target;
2. mean final MSE is at least 20% lower than random;
3. mean final MSE is no more than 1.50× exact-pixel greedy;
4. every checkpoint/hash, deterministic replay, candidate uniqueness, finiteness, and no-retraining integrity check passes.

It is not required to outperform the learned-pixel planner. That comparison is reported, not used to redefine success.

## Secondary outcomes

- best-frame pixel MSE and best step;
- final and best relative improvements;
- exact-pixel candidate top-1/top-5, mean rank, and regret for each learned method;
- `latent_ranking` versus `latent_mse` candidate ranking and trajectory differences;
- per-step latent score and exact pixel outcome;
- Spearman association between latent scores and exact candidate pixel scores;
- runtime;
- qualitative montage and trajectory plots.

## Interpretation rules

- Success would show that the formal one-step latent result transfers to this fixed multi-step candidate-selection setup.
- Failure would show a gap between one-step counterfactual discrimination and long-horizon target planning; it would not invalidate the formal result.
- A ranking-aware advantage over MSE-only is informative but not required for the primary deployment classification.
- Results apply only to the frozen 64×64 grayscale straight-stroke setting and target-guided proposal process.
- The method is JEPA-inspired, not a canonical JEPA.

## Hard prohibitions

- no autoencoder or predictor retraining/fine-tuning;
- no checkpoint substitution after hash freeze;
- no use of formal test/stress results to select a seed;
- no proposal, score, renderer, step, candidate, threshold, or seed changes after this freeze;
- no controlled rerun or retuning;
- no deletion of failed incomplete outputs without written adjudication;
- no claim that four-way retrieval already proved multi-step painting.
