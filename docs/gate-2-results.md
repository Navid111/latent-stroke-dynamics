# Gate 2 formal results — deterministic one-step latent prediction

**Formal run date:** 2026-08-21  
**Status:** Complete  
**Decision:** **Fail under the frozen Gate 2 rule**  
**Selected family:** MLP, selected by mean validation action-region MSE  
**Formal eligibility:** True

## Executive result

The predictor learned the average one-stroke latent consequence extremely well, generalized consistently across initialization seeds and stress slices, and beat both trivial baselines at every primary crowding level. It nevertheless failed the action-grounding requirement: exact four-way counterfactual retrieval reached only 27.7%, versus a frozen 50% threshold and a 25% random-choice reference.

This is a mixed scientific result, not an implementation failure. One-step latent residuals are predictable under average error, but the current deterministic patch-wise predictors are not precise enough for action-level planning.

## Integrity checks

- The exact formal configuration was committed before formal data were generated.
- Untouched amended seeds `20260824`–`20260829` were used.
- Model seeds were `11`, `22`, and `33`.
- All rendered and encoded counterfactual candidates were unique.
- All metrics were finite.
- The tiny-overfit sanity check reduced loss by 97.7%.
- Every learned seed beat identity on action-region MSE.
- `formal_run_requested` and `formal_eligible` were both true.

The plotting `tight_layout` warning was cosmetic and did not affect data, models, metrics, or the decision.

## Primary test result

| Predictor | Action-region MSE | Improvement vs identity | Improvement vs mean delta |
|---|---:|---:|---:|
| Identity | 0.002250 | — | — |
| Mean delta | 0.002003 | 11.0% | — |
| Selected MLP, three-seed mean | 0.000860 | **61.8%** | **57.1%** |

The selected MLP test action-region MSE by seed was:

- seed 11: 0.000861;
- seed 22: 0.000841;
- seed 33: 0.000877.

The narrow spread confirms stable initialization behavior.

## Crowding result

| Prior strokes | Improvement over identity |
|---:|---:|
| 0 | **79.0%** |
| 5 | **43.3%** |
| 15 | **25.0%** |

Improvement remained positive at every frozen primary crowding level.

## Counterfactual retrieval

The selected MLP family averaged **27.7%** top-1 retrieval across its three seeds. This is only modestly above the 25% random-choice reference, below the 35% lower boundary named in the frozen fail rule, and far below the required 50% threshold.

Development diagnostics had already identified width-changed outcomes as the dominant confusion. The larger formal training set improved average prediction substantially but did not resolve exact action discrimination.

## Frozen decision table

| Criterion | Requirement | Formal result | Outcome |
|---|---|---:|---|
| Error vs identity | ≥30% improvement | 61.8% | Pass |
| Error vs mean delta | ≥30% improvement | 57.1% | Pass |
| Every crowding level | Positive improvement | All positive | Pass |
| Counterfactual retrieval | ≥50% | 27.7% | **Fail** |
| Sanity and seed stability | No failure/collapse | Passed | Pass |

Because all criteria were conjunctive, the retrieval failure makes the overall Gate 2 decision **fail**. The result must not be relabeled borderline: the frozen protocol defines retrieval at or below 35% as fail.

## Secondary stress slices

The MLP three-seed mean action-region MSE remained substantially below identity on all secondary slices:

| Stress slice | MLP MSE | Identity MSE | Approx. improvement |
|---|---:|---:|---:|
| Unseen width 5 | 0.001028 | 0.002547 | 59.7% |
| Unseen intensities | 0.000902 | 0.002466 | 63.4% |
| Unseen crowding 10 | 0.000916 | 0.001342 | 31.7% |

This reinforces the distinction between strong average-error generalization and weak exact counterfactual ranking.

## Interpretation

The most defensible conclusion is:

1. frozen DINOv2 patch-token changes after a deterministic stroke are highly predictable on average;
2. the learned model captures useful position, intensity, and crowding-dependent structure;
3. MSE prediction smooths or underestimates action-specific detail, especially stroke width;
4. low average latent error alone is insufficient evidence that a model can rank closely related actions for planning.

Gate 3 target-guided planning must not begin with this predictor under the project’s frozen gate logic.

## Next work

1. Preserve and archive the completed formal result without rerunning it.
2. Run the existing no-retraining retrieval decomposition on the formal output for thesis interpretation only; it cannot change the decision.
3. Implement the preregistered small action-conditioned pixel-space control.
4. Compare latent and pixel-space behavior, then write the thesis result as a mixed feasibility finding.
5. Treat contrastive retrieval losses, spatially interacting predictors, and width-specific objectives as future work or clearly labeled post-formal ablations—not as replacements for this formal result.
