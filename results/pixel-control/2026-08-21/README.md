# Paired pixel-space control — 2026-08-21

**Eligibility:** True  
**Status:** Success  
**Selected family:** MLP  
**Four-way retrieval:** 100% across all three seeds

The 833-parameter MLP reduced action-region pixel MSE by 99.950% versus identity and 99.948% versus mean delta. It retrieved the exact rendered outcome for every one of the 300 paired test examples under every seed, including perfect width discrimination. The exact compositing oracle also achieved 100% retrieval and effectively zero error.

The paired control establishes that exact action information is recoverable in the full-resolution pixel formulation. It does not revise the recorded latent Gate 2 fail; it localizes that failure to the overall tested latent patch formulation.

See [`docs/pixel-control-results.md`](../../../docs/pixel-control-results.md).
