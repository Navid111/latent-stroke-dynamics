# Ranking-aware latent formal result

**Decision:** `formal_ranking_rescue_success`  
**Run:** single authorized formal execution  
**Elapsed:** 152.96 seconds  
**Rerun or retuning:** forbidden

## Primary result

| Method | Exact-action retrieval | Action-region MSE | Mean true margin |
|---|---:|---:|---:|
| MSE-only | 31.44% | 0.483698 | -0.002585 |
| Ranking-aware | **74.44%** | 0.491221 | **0.001862** |

Ranking-aware training improved retrieval by **43.00 percentage points**. Its average error was only 1.56% higher than the MSE-only predictor, while remaining 70.33% below identity and 68.24% below mean delta. Thus the ranking objective traded a very small amount of average MSE for a large gain in action discrimination.

Every preregistered success condition passed: retrieval at least 50%, at least a 10-point gain over matched MSE-only, at least 30% improvement over identity and mean delta, positive improvement at crowding 0/5/15, all ranking seeds beating identity, 100% oracle retrieval, and full implementation integrity.

## Primary-crowding reductions versus identity

- crowding 0: 73.89%;
- crowding 5: 69.40%;
- crowding 15: 64.80%.

## Secondary stress results

| Stress split | MSE-only retrieval | Ranking-aware retrieval | Gain |
|---|---:|---:|---:|
| Unseen width 5 | 62.33% | **87.33%** | +25.00 points |
| Unseen intensities | 33.67% | **74.67%** | +41.00 points |
| Crowding 30 | 46.33% | **57.67%** | +11.33 points |
| Crowding 60 | 36.33% | **43.00%** | +6.67 points |

The ranking benefit generalized strongly to unseen width and intensity and remained positive under heavier crowding. Absolute retrieval fell below 50% at crowding 60, which is an important boundary rather than a failure of the primary result.

## Scientific interpretation

The earlier task-autoencoder failure was not explained by complete loss of stroke information. A substantial part of the failure came from objective mismatch: minimizing average latent residual error produced plausible average changes but weak counterfactual action ordering. Adding an explicit candidate-ranking loss converted the same frozen representation, same predictor family, and same data-generating process from 31.44% to 74.44% formal retrieval.

The justified claim is specific: **within the tested frozen task-autoencoder representation, action-discriminative information was present and became usable when the predictor objective was aligned with counterfactual action selection.** This does not establish that all latent encoders work, and the model is JEPA-inspired rather than a canonical JEPA.

## Planning implication

The formal gate permits a separately frozen latent-space candidate-selection painter. That planner must be treated as a new deployment experiment, use the existing formal checkpoints without retraining, select actions by predicted next-latent distance to a frozen target latent, and be evaluated without altering this formal result.
