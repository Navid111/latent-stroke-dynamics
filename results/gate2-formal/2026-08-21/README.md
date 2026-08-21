# Gate 2 formal snapshot — 2026-08-21

**Decision:** Fail under the frozen rule.  
**Formal eligibility:** True.  
**Selected family:** MLP.  
**Reason:** Counterfactual retrieval was 27.7%, below the frozen 50% requirement and 35% fail boundary.

The selected three-seed MLP nevertheless improved test action-region MSE by 61.8% versus identity and 57.1% versus mean delta, remained positive at all crowding levels, passed all implementation checks, and generalized strongly to all three stress slices.

See [`docs/gate-2-results.md`](../../../docs/gate-2-results.md) for the complete interpretation and [`docs/gate-2-formal-config.md`](../../../docs/gate-2-formal-config.md) for the frozen command.

This directory initially archives the decisive configuration and gate-diagnostic artifacts supplied from the completed local run. Generated feature caches and raw model outputs remain ignored by Git.
