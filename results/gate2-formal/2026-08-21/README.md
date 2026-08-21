# Gate 2 formal snapshot — 2026-08-21

**Decision:** Fail under the frozen rule.  
**Formal eligibility:** True.  
**Selected family:** MLP.  
**Reason:** Counterfactual retrieval was 27.7%, below the frozen 50% requirement and 35% fail boundary.

The selected three-seed MLP improved test action-region MSE by 61.8% versus identity and 57.1% versus mean delta, remained positive at all crowding levels, passed all implementation checks, and generalized strongly to all three stress slices.

The post-hoc retrieval decomposition showed stable MLP accuracy across seeds (26.3–29.3%; standard deviation 1.53 percentage points). It selected width-changed outcomes 48.2% of the time. True outcomes beat shifted-position and changed-intensity alternatives 77.9% and 75.2% of the time, but beat changed-width alternatives only 40.7% of the time.

See:

- [`docs/gate-2-results.md`](../../../docs/gate-2-results.md)
- [`docs/gate-2-formal-visual-review.md`](../../../docs/gate-2-formal-visual-review.md)
- [`docs/gate-2-formal-retrieval-diagnostics.md`](../../../docs/gate-2-formal-retrieval-diagnostics.md)
- [`docs/gate-2-formal-config.md`](../../../docs/gate-2-formal-config.md)

This directory archives the decisive configuration, gate diagnostic, and compact formal retrieval summaries supplied from the completed local run. Feature caches and the full raw output directory remain ignored by Git.
