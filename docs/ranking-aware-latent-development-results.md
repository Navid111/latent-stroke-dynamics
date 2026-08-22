# Ranking-aware latent development result — raw review

**Run:** single authorized development grid  
**Elapsed:** 86.27 seconds  
**Formal data:** untouched and unauthorized  
**Rerun:** forbidden

## Main signal

Validation-only selection chose:

```text
ranking weight = 1.0
temperature = 0.05
```

| Split | MSE-only retrieval | Ranking-aware retrieval | Absolute gain |
|---|---:|---:|---:|
| Validation | 28.13% | 70.83% | 42.71 points |
| Diagnostic test | 27.08% | 76.04% | 48.96 points |

The diagnostic action-region MSE changed only from `0.621324` to `0.623307`, a roughly 0.32% increase, while retrieval nearly tripled. Mean true margin changed from negative (`-0.002854`) to positive (`0.001664`).

On the diagnostic split, true-versus-intensity wins improved from 43.23% to 89.06%, and true-versus-width wins improved from 64.58% to 83.85%. Position remained 100%.

These metrics are strongly encouraging but remain development-only. They do not authorize a latent planner or support a formal claim.

## Raw integrity flag

The raw summary marked integrity false only through the combined field `all_metrics_and_histories_finite`. The runner concatenated two valid but heterogeneous history schemas:

- MSE-only rows have balanced-MSE history fields;
- ranking-aware rows additionally have total and ranking-cross-entropy fields.

Pandas filled the ranking-only columns on MSE-only rows with structural `NaN` values. The raw whole-numeric-table finite check then treated those inapplicable blanks as non-finite losses.

All displayed metrics are finite, every oracle passed at 100%, candidates are unique, parameter counts match, and the ranking overfit objective decreased by 39.13%. A no-rerun adjudicator now checks common history columns on every row and ranking-specific columns only on ranking-aware rows. The raw summary remains unchanged.

## Pending boundary

Run and validate the JSON/CSV-only adjudicator before freezing the selected setting or implementing the formal command. Do not rerun development and do not generate formal seeds.
