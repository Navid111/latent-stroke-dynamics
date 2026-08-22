# Ranking-aware latent dynamics follow-up protocol — 2026-08-22

**Status:** Frozen before implementation and before follow-up data generation  
**Role:** Post-core exploratory latent rescue study  
**Deadline:** 2026-09-24  
**Historical Gate 2, pixel-control, Stage 3, and representation-extension decisions:** Unchanged

## 1. Research question

The completed representation extension showed that the frozen task-autoencoder latent supported strong average next-state prediction but only 37.89% four-way action retrieval. The main mismatch was therefore between average latent regression and planning-relevant candidate ranking.

This follow-up asks:

> When the encoder, action representation, predictor architecture, and data distribution are held fixed, can adding an explicit counterfactual ranking objective improve exact next-action retrieval while preserving average latent prediction quality?

This is an action-conditioned latent dynamics experiment. It is not a new JEPA architecture and does not jointly train the encoder.

## 2. Frozen representation

Use the selected task autoencoder from the single completed representation-extension run:

- architecture: `StrokeAutoencoder`;
- latent shape: 32×16×16;
- total parameters: 49,569;
- selected seed: 101;
- selected epoch: 50;
- checkpoint SHA-256: `95de3ecef8eeb7a350e862fa21185a168d9304870cb0c8391cbd008e88d93900`;
- local checkpoint: `outputs/representation-extension-2026-08-22/task_autoencoder/checkpoints/task_autoencoder.pt`;
- encoder weights: frozen throughout.

Reuse the saved full-run latent channel statistics at:

```text
outputs/representation-extension-2026-08-22/task_autoencoder/latent_channel_statistics.json
```

Before any follow-up data are generated, validation-only mode must record that file's SHA-256. A separate commit must freeze the observed hash. Development is unauthorized until that hash is committed.

## 3. Fixed transition and counterfactual setup

- canvas: 64×64 grayscale;
- renderer: existing deterministic straight-line renderer;
- primary crowding: `{0, 5, 15}`;
- widths: `{1, 2, 3, 4}`;
- intensities: `{0, 32, 64, 96, 128}`;
- minimum normalized length: `0.20`;
- action: existing seven-value stroke vector plus fractional 16×16 action coverage and token coordinates;
- four candidates: true, shifted position, changed width, changed intensity;
- candidate scoring: normalized-feature MSE over the fixed union of candidate action regions;
- chance retrieval: 25%.

The representation and candidate construction are fixed. No new encoder, canvas size, stroke family, or renderer is included in this experiment.

## 4. Development-only selection data

| Split | Examples | Seed | Use |
|---|---:|---:|---|
| Train | 128 | `20261101` | Fit development predictors |
| Validation | 64 | `20261102` | Select ranking hyperparameters only |
| Diagnostic test | 64 | `20261103` | Post-selection development diagnostic only |

Candidate sets are generated for all three development splits because ranking loss requires train and validation counterfactuals. Diagnostic-test rows cannot affect any setting.

Development may select only the ranking-loss weight and temperature from the frozen grid below. It cannot change architecture, data distribution, epochs, thresholds, or formal sizes.

## 5. Reserved untouched formal data

These seeds must remain ungenerated until development is archived and one exact formal command is committed.

| Split | Examples | Seed |
|---|---:|---:|
| Train | 1,000 | `20261104` |
| Validation | 200 | `20261105` |
| Test | 300 | `20261106` |
| Unseen width 5 | 100 | `20261107` |
| Unseen intensities `{16, 80, 176}` | 100 | `20261108` |
| Crowding 30 | 100 | `20261109` |
| Crowding 60 | 100 | `20261110` |

All follow-up fingerprints must be disjoint from one another. The follow-up is additional evidence and never replaces earlier test results.

## 6. Matched models

The predictor architecture is fixed to the existing shared MLP:

- input: current 32-dimensional token, seven action values, one action-mask value, and x/y token coordinates;
- hidden dimension: 256;
- output: 32-dimensional latent residual;
- trainable parameters: 19,232;
- model seeds: `11`, `22`, `33`;
- optimizer: AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- batch size: 16;
- maximum epochs: 30;
- patience: 6;
- CPU only.

Compare:

1. **MSE-only matched baseline:** existing balanced residual MSE.
2. **Ranking-aware predictor:** the same MLP and initialization budget, trained with balanced residual MSE plus counterfactual ranking cross-entropy.

Identity and training mean-delta baselines remain evaluation anchors.

## 7. Ranking objective

For every train/validation transition, encode the four exact candidate next canvases. Given predicted next tokens, compute the same normalized-feature candidate scores used by retrieval:

```text
s_j = masked_normalized_feature_MSE(z_hat_next, candidate_j)
```

Candidate zero is true. Ranking logits and loss are:

```text
logit_j = -s_j / temperature
L_rank = cross_entropy(logits, target=0)
L_total = L_balanced_MSE + lambda * L_rank
```

Development grid:

- `lambda ∈ {0.1, 0.3, 1.0}`;
- `temperature ∈ {0.05, 0.1}`.

Each of the six settings uses all three model seeds. Select one setting using, in order:

1. highest mean validation four-way retrieval;
2. highest mean validation true margin;
3. lowest mean validation action-region MSE;
4. lower lambda;
5. higher temperature.

No diagnostic-test value may enter selection. The selected pair must be committed before formal data generation.

For each seed, early stopping uses its validation total objective, not top-1 test retrieval.

## 8. Formal comparison and decision

The later formal run will train exactly:

- three MSE-only MLPs;
- three ranking-aware MLPs using the committed development-selected `lambda` and `temperature`.

Primary ranking-aware success requires all of:

1. at least 50% four-way test retrieval;
2. at least a 10 percentage-point absolute retrieval improvement over the matched MSE-only family mean;
3. at least 30% action-region MSE improvement over identity;
4. at least 30% action-region MSE improvement over training mean delta;
5. positive improvement over identity at crowding 0, 5, and 15;
6. 100% exact-target oracle retrieval;
7. unique encoded candidates, finite metrics, all seeds beating identity on average error, and a decreasing tiny-overfit loss.

If retrieval improves but remains below 50%, report a partial ranking improvement, not an action-usable latent. If it fails to improve meaningfully, preserve the negative result.

Only a formal primary success authorizes a separate latent-planning protocol. Development results never authorize planning.

## 9. Required artifacts

Development must save:

- exact config and environment metadata;
- checkpoint and latent-statistics hashes;
- split fingerprints and disjointness result;
- per-setting and per-seed training histories;
- validation selection table;
- diagnostic-test prediction and retrieval tables;
- pairwise retrieval and candidate-frequency diagnostics;
- oracle, uniqueness, finite-value, parameter, leakage, and overfit checks;
- one development summary clearly marked non-formal.

Formal artifacts will be specified in a separate frozen command after development review.

## 10. Stop rules and scope protection

1. Commit this protocol/config before implementation and data generation.
2. Implement validation-only mode and tests.
3. Record and freeze the saved latent-statistics hash before development.
4. Run one development command and archive every outcome.
5. Freeze the selected ranking setting and formal command in separate commits.
6. Run the formal comparison once; never tune or rerun from its outcome.

Forbidden during this follow-up:

- encoder retraining or fine-tuning;
- changing the autoencoder checkpoint;
- additional encoders;
- larger canvases;
- new stroke primitives or renderers;
- selecting on diagnostic-test or formal-test results;
- revising any completed historical decision.

Canvas-size, stroke-family, renderer, higher-resolution latent, and true JEPA-inspired studies remain possible later, but they require separate protocols so that this experiment isolates the ranking objective.
