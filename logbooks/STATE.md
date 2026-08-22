# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Post-formal latent planner design  
**Status:** Formal ranking rescue succeeded; formal run permanently closed

## Formal result

- classification: `formal_ranking_rescue_success`;
- ranking-aware retrieval: 74.44%;
- MSE-only retrieval: 31.44%;
- absolute gain: 43.00 percentage points;
- ranking-aware action-region MSE: 0.491221;
- MSE-only action-region MSE: 0.483698;
- improvement versus identity: 70.33%;
- improvement versus mean delta: 68.24%;
- all primary crowding improvements positive;
- all ranking seeds beat identity;
- every oracle: 100%;
- implementation integrity: passed.

## Stress boundary

Ranking-aware retrieval was 87.33% for unseen width 5, 74.67% for unseen intensities, 57.67% at crowding 30, and 43.00% at crowding 60. Heavy crowding remains a limitation.

## Scientific conclusion

The tested task latent contained usable stroke-action information, but MSE-only training did not align it with counterfactual action ranking. The frozen ranking loss made that information operational without changing the representation or predictor architecture.

## Next action

Freeze a separately scoped latent-space planner protocol using existing formal ranking-aware checkpoints and no retraining. Then implement validation-only support before any planner evaluation.
