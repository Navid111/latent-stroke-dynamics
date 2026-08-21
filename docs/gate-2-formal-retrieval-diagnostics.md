# Gate 2 formal retrieval decomposition — 2026-08-21

**Scope:** Post-hoc interpretation of the completed formal output  
**Formal decision:** Remains **fail**  
**Selected family:** MLP, selected earlier by validation MSE  
**Scientific settings changed:** None

## Validation of the diagnostic

The revised multi-seed-aware diagnostic passed as part of the complete local suite:

```text
19 passed in 1.52s
```

It averages family behavior over the three preregistered seeds without treating repeated predictions on the same 300 test examples as 900 independent trials. It does not select a seed or model using formal retrieval.

## Seed stability

| MLP seed | Correct | Accuracy | 95% Wilson interval |
|---:|---:|---:|---:|
| 11 | 79/300 | 26.3% | 21.7–31.6% |
| 22 | 82/300 | 27.3% | 22.6–32.6% |
| 33 | 88/300 | 29.3% | 24.5–34.7% |

Family mean retrieval was **27.7%**, with a seed standard deviation of only **1.53 percentage points** and a range of 26.3–29.3%. No seed approached the 50% gate threshold; even the upper Wilson bound of the strongest seed was 34.7%.

The retrieval failure is therefore stable rather than an unlucky initialization.

## Which outcomes the selected MLP chose

Averaged across seeds, the MLP selected:

- true outcome: **27.7%**;
- shifted position: **11.9%**;
- changed width: **48.2%**;
- changed intensity: **12.2%**.

Nearly half of all selections were width-changed alternatives. Position and intensity alternatives were selected relatively rarely.

## Pairwise decomposition

| Comparison | True-outcome win rate |
|---|---:|
| True vs shifted position | **77.9%** |
| True vs changed width | **40.7%** |
| True vs changed intensity | **75.2%** |

This strongly rejects a broad action-insensitivity explanation. The model usually distinguishes position and intensity changes, but systematically struggles with exact width.

The width-alternative score minus true score had mean `-2.60e-05` and median `-3.26e-05`. Because lower scores are preferred, the negative values mean the changed-width outcome was slightly closer to the prediction on average. By contrast, the position and intensity gaps were clearly positive.

## Relationship to development v2

More formal training data substantially improved average MSE, position discrimination, and intensity discrimination. It did not repair width:

- development MLP true-vs-width win rate: 40.6%;
- formal MLP true-vs-width win rate: 40.7%.

This near-identical result is especially informative. The persistent failure is not explained by the smaller development set, insufficient epochs, or a single unlucky seed.

## Final interpretation

The learned predictor is action-conditioned and preserves substantial action information. Its failure is localized: deterministic MSE prediction of DINOv2 patch-token residuals smooths or miscalibrates the magnitude and spatial extent needed to distinguish nearby stroke widths.

The formal thesis claim should therefore be narrower than “latent prediction does not work”:

> Frozen DINOv2 spatial features support accurate average one-step stroke dynamics, but low latent MSE does not guarantee sufficiently precise discrimination among closely related stroke actions, with stroke width emerging as the dominant failure mode.

No further latent-model tuning may replace or revise the recorded formal result. The preregistered pixel-space action-conditioned control is the next experiment.
