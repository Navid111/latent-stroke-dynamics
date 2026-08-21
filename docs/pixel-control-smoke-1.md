# Pixel-space control smoke 1 — 2026-08-21

**Scope:** Development-only engineering smoke  
**Data:** 128 train / 32 validation / 64 test  
**Data seeds:** `20260830`, `20260831`, `20260832`  
**Model seed:** `11`  
**Decision authority:** None; `diagnostic_only`

## Test and integrity checks

The complete local suite passed:

```text
24 passed in 2.06s
```

All required smoke integrity checks passed:

- overfit loss decreased by 89.6%;
- all metrics were finite;
- all four rendered candidates were unique;
- exact compositing oracle top-1 retrieval was 100%;
- exact-oracle maximum action-region MSE was `3.55e-15`;
- implementation sanity passed.

The oracle result verifies the normalized-pixel target, exact action mask, clamped reconstruction, candidate ordering, and retrieval scoring agree with the deterministic renderer.

## Validation behavior

Validation selected the MLP. Both learned curves decreased throughout the 12 smoke epochs; the MLP finished with substantially lower balanced validation residual MSE than the linear model. There was no divergence or early-stopping anomaly.

## Development-only results

The selected MLP achieved:

- test action-region MSE: `0.04605`;
- improvement versus identity: 91.1%;
- improvement versus mean delta: 90.7%;
- four-way retrieval: 93.75%;
- crowding improvements: 93.8%, 92.0%, and 86.4% at crowding 0, 5, and 15.

Candidate selections were:

- true: 93.75%;
- shifted position: 0%;
- changed width: 1.56%;
- changed intensity: 4.69%.

Pairwise true-outcome win rates were:

- versus shifted position: 100%;
- versus changed width: 98.4%;
- versus changed intensity: 95.3%.

The linear model reached only 42.2% four-way retrieval, despite strong position discrimination, so the nonlinear compositing interaction is consequential rather than decorative.

## Visual review

The stacked selection plot clearly shows the MLP dominated by true selections and the oracle at 100%. The pairwise plot shows the MLP near the oracle for all three alternative types, including width. Legends are outside the plotting area and do not obscure bars.

## Interpretation and authorization

This smoke provides strong engineering evidence that exact action information is recoverable in the full-resolution pixel formulation. It does not establish the paired-control result because it uses development-only data and one model seed.

No implementation repair or hyperparameter change is justified. The previously frozen paired configuration is authorized to run once. The latent Gate 2 fail remains unchanged.
