# Stage 3 random/exact smoke 1 review — 2026-08-22

**Status:** Engineering pass  
**Role:** Development-only behavior and artifact check  
**Formal Stage 3 decision:** Not made

## Numerical result

| Method | Initial MSE | Final MSE | Relative improvement | Final MAE | Improving steps |
|---|---:|---:|---:|---:|---:|
| Random | 0.149447 | 0.144913 | 3.03% | 0.221286 | 11/20 |
| Exact greedy | 0.149447 | 0.074427 | 50.20% | 0.134931 | 20/20 |

After the same 20-stroke and 32-candidate budgets, exact greedy final MSE was approximately 48.6% below random and final MAE was approximately 39.0% below random.

## Integrity checks

- deterministic exact replay passed;
- all metrics were finite;
- exact final MSE was below random;
- the run correctly remained diagnostic only;
- no earlier formal result was touched.

## Visual review

The random output captures little of the target’s precise geometry. Its MSE curve oscillates near the initial value and ends with only a small net improvement.

The exact-greedy output recovers several major target structures, including the dominant dark diagonal and broad upper/left strokes. It remains incomplete after only 20 actions, as expected, but every selected step lowers target MSE. The exact curve is smooth and strictly decreasing.

Both absolute-error panels use the stated fixed 0–255 color scale, so their relative brightness is directly comparable and does not repeat the auto-scaling issue found in the earlier pixel example figure. The exact panel has visibly less high-intensity error.

## Decision

The target preprocessing, proposal policy, exact execution, deterministic planning loop, metrics, and artifact outputs behave coherently. No engineering repair is required before adding the separate learned-pixel checkpoint and candidate-ranking path.
