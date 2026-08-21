# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Experimental core complete  
**Status:** Paired pixel-space control succeeded; thesis comparison and writing next

## Final experimental results

### Gate 1 — representation sensitivity

**Pass.** Frozen DINOv2-small patch features preserve localized one-stroke changes.

### Gate 2 — latent one-step prediction

**Formal fail.** The selected MLP improved action-region latent MSE by 61.8% versus identity and 57.1% versus mean delta, but four-way exact-action retrieval was only 27.7%. Width was the dominant failure: 40.7% true-vs-width pairwise accuracy.

### Paired pixel-space control

**Success.** The validation-selected 833-parameter MLP achieved:

- action-region pixel MSE: 0.000249;
- improvement versus identity: 99.950%;
- improvement versus mean delta: 99.948%;
- four-way exact-action retrieval: 100% for every seed;
- position, width, and intensity pairwise accuracy: all 100%;
- positive performance at all crowding levels;
- strong performance on all three stress slices;
- exact-oracle retrieval: 100%;
- all implementation checks passed.

The latent Gate 2 decision remains fail. The pixel result localizes the bottleneck to the overall tested latent patch formulation rather than the transition data, action information, candidate construction, or general learnability of the deterministic dynamics.

## Next actions

1. Review and archive the remaining pixel-control plots.
2. Write the paired latent-versus-pixel methods and results comparison.
3. Draft limitations carefully: the pixel control also changes spatial resolution and uses an exact full-resolution mask, so it does not isolate DINOv2 alone.
4. Prepare thesis figures and result tables.
5. Do not begin Gate 3 planning unless the thesis scope is explicitly reopened after the core write-up.

## Boundaries

- Do not rerun or retune either completed paired experiment.
- Do not revise the latent Gate 2 fail.
- Do not present the pixel control as a novel painter or JEPA architecture.
- Treat higher-resolution latent features, contrastive objectives, spatial interaction, and width-aware losses as future work.
