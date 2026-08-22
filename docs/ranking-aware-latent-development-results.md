# Ranking-aware latent development result — final review

**Run:** single authorized development grid  
**Elapsed:** 86.27 seconds  
**Adjudication:** Passed after 69 tests  
**Formal data:** untouched and unauthorized  
**Rerun:** forbidden

## Selected setting

Validation-only selection froze:

```text
ranking weight = 1.0
temperature = 0.05
```

| Split | MSE-only retrieval | Ranking-aware retrieval | Absolute gain |
|---|---:|---:|---:|
| Validation | 28.13% | 70.83% | 42.71 points |
| Diagnostic test | 27.08% | 76.04% | 48.96 points |

The diagnostic action-region MSE changed only from `0.621324` to `0.623307`, a roughly 0.32% increase, while retrieval nearly tripled. Mean true margin changed from `-0.002854` to `0.001664`.

On the diagnostic split, true-versus-intensity wins improved from 43.23% to 89.06%, true-versus-width wins improved from 64.58% to 83.85%, and position remained 100%.

## Integrity adjudication

The raw false flag came from a whole-table finiteness check over heterogeneous history schemas. All 90 MSE-only history rows had finite applicable fields. All 540 ranking-aware history rows had finite common and ranking-specific fields. Ranking-only fields on MSE-only rows were expected structural blanks, not loss values.

All prediction/retrieval metrics were finite, every protocol oracle passed, all encoded candidates were unique, parameter counts matched, validation alone selected the setting, and the tiny-overfit objective decreased. Written-protocol implementation integrity passed without data/model execution or scientific metric recomputation.

## Interpretation boundary

This development result strongly supports the hypothesis that objective mismatch, rather than complete latent information loss, contributed to prior retrieval failure. It is not a formal claim. The frozen setting must now be tested once on untouched formal seeds.
