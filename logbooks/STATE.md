# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Pixel-space explanatory control  
**Status:** Protocol frozen before implementation

## Closed results

### Gate 1

Formal pass: frozen DINOv2-small patch features preserve localized stroke changes.

### Gate 2

Formal fail under the frozen conjunctive rule:

- selected MLP action-region MSE: 0.000860;
- improvement versus identity: 61.8%;
- improvement versus mean delta: 57.1%;
- crowding improvements: +79.0%, +43.3%, and +25.0%;
- counterfactual retrieval: 27.7%;
- all implementation and seed checks passed.

Formal retrieval was stable across seeds and failed specifically on width: the selected MLP chose width-changed outcomes 48.2% of the time and beat them pairwise only 40.7% of the time. Gate 2 is closed; no reruns or latent test tuning are authorized.

## Active question

Can a minimal action-conditioned predictor recover exact stroke outcomes in normalized pixel space when evaluated on paired deterministic transitions?

## Frozen pixel-control design

- Same paired 1,000/200/300 transition splits and three stress slices as Gate 2.
- Normalized 64×64 grayscale pixels and residual prediction.
- Per-pixel input: current value, seven-value action vector, exact proposed-action mask, and x/y coordinates.
- Identity, mean-delta, shared linear, shared `11 -> 64 -> 1` MLP, and exact compositing oracle.
- Balanced inside/outside pixel residual MSE.
- Model seeds `11`, `22`, and `33`.
- Validation-only family selection.
- Same 50% four-way retrieval standard for explanatory success.

See `docs/pixel-space-control-protocol.md`.

## Next actions

1. Implement the smallest pixel-control core and deterministic tests.
2. Add an end-to-end CPU experiment with saved tables and plots.
3. Run the 128/32/64 development-only smoke on seeds `20260830`–`20260832`.
4. If implementation checks pass, run the paired control once.
5. Compare pixel and latent width discrimination in the thesis.

## Boundaries

- Do not rerun or revise the latent Gate 2 formal result.
- Do not begin Gate 3 planning.
- Do not add contrastive, spatially interacting, or width-specific latent objectives to the primary result.
- The pixel control is explanatory and cannot retroactively pass Gate 2.
