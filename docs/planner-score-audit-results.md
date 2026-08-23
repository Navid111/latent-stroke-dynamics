# Planner-score development audit results

## Status

The single authorized development score audit completed successfully and is now closed. Implementation integrity passed, all models stayed frozen, no model was trained or fine-tuned, no closed target was reused, and all historical results remain unchanged.

## Design

The audit evaluated 10 frozen predictor/score pairs over 72 independently defined candidate sets: eight new targets, nine states per target, and 128 unique candidates per state. Exact pixel MSE after rendering supplied the candidate-order labels. Selection followed the preregistered lexicographic rule, beginning with lowest mean exact regret.

## Frozen selection

The selected pair is:

- predictor family: `mse_only`;
- planner score: `normalized_latent_l1`;
- mean exact regret: `0.0010517144069821269`;
- exact top-5 rate: `0.5138888888888888`;
- mean score-to-exact Spearman: `0.6192424342849114`.

Compared with the same MSE-only predictors under the previous normalized-latent MSE score, normalized-latent L1 reduced mean exact regret by about 28.8%, increased top-5 selection from 33.33% to 51.39%, improved mean exact rank from 29.46 to 12.39, and increased Spearman from 0.582 to 0.619.

The pixel-error-weighted MSE score had the highest top-1 rate within the MSE-only family at 20.83%, but its mean exact regret was higher than L1. Decoded pixel L1 produced higher Spearman, including 0.681 with the ranking-aware ensemble, but again had higher regret. The frozen primary criterion therefore selects MSE-only plus normalized-latent L1 without a tie-break.

## Interpretation

The result supports the score-misalignment diagnosis: changing only the planner's reference score improved candidate selection materially without changing or retraining the dynamics model. It also reinforces that the ranking-aware predictor's formal four-way retrieval advantage did not translate into the strongest 128-way planner scoring in this development study.

This is not yet evidence of better sequential painting. The next separately guarded phase must compare the selected L1 score in actual 100-step planning, both forced-horizon and with the preregistered zero-margin no-op rule, against current latent MSE, exact pixel, and learned pixel on three different reserved targets.

## Immutable boundaries

- Do not rerun or tune against this development audit.
- Keep the selected pair fixed as MSE-only plus normalized-latent L1.
- Keep the Sobel coefficient and all rejected candidate scores unchanged.
- Do not authorize planner development until its guarded runner passes validation.
- Confirmatory targets remain untouched and unauthorized.
