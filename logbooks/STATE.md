# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Thesis comparison and writing  
**Status:** Experimental core complete; final figures reviewed

## Final experimental results

### Gate 1

**Pass.** Frozen DINOv2-small patch features preserve localized one-stroke changes.

### Latent Gate 2

**Formal fail.** Average latent prediction was strong, but four-way exact-action retrieval was 27.7%. Width was the dominant failure at 40.7% pairwise true-vs-width accuracy.

### Paired pixel control

**Success.** The 833-parameter MLP improved action-region pixel MSE by 99.950% versus identity and retrieved the true outcome for all 300 test examples under all three seeds. Position, width, and intensity pairwise accuracy were all 100%.

## Figure review

Final pixel figures agree with the saved tables. Training is stable, retrieval is unambiguous, and crowding robustness is near complete. The example prediction is visually indistinguishable at ordinary scale. Its error panel is auto-scaled without a colorbar and must be labeled carefully or omitted from the thesis figure because it visually amplifies tiny errors.

## Writing artifacts

- `docs/latent-vs-pixel-comparison.md`
- `docs/thesis-results-draft.md`
- `docs/pixel-control-formal-visual-review.md`

## Next actions

1. Convert the results draft into the thesis chapter structure.
2. Add verified literature citations and figure numbers.
3. Create a thesis-ready latent-versus-pixel comparison table and select final plots.
4. Write methods while exact implementation details remain fresh.
5. Draft discussion, limitations, and conclusion.

## Boundaries

- Do not rerun or retune either paired experiment.
- Do not revise the latent Gate 2 fail.
- Do not compare raw latent and pixel MSE values directly across spaces.
- Do not present the pixel control as a complete painter or a novel JEPA model.
- Do not begin Gate 3 planning before the core thesis write-up is complete and scope is explicitly reconsidered.
