# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Pixel-space explanatory control  
**Status:** Smoke passed; single frozen paired run authorized

## Closed results

- Gate 1 formally passed.
- Latent Gate 2 formally failed because retrieval was 27.7%, despite strong and stable average-error prediction.
- Formal Gate 2 diagnosis isolated width as the main failure: 40.7% pairwise true-vs-width and 48.2% width-alternative selections.

No latent rerun or test tuning is authorized.

## Pixel-control smoke

The development-only 128/32/64 smoke completed after all 24 tests passed.

- validation-selected family: MLP;
- action-region improvement versus identity: 91.1%;
- action-region improvement versus mean delta: 90.7%;
- four-way retrieval: 93.75%;
- true-vs-width pairwise win rate: 98.4%;
- exact oracle retrieval: 100%;
- exact oracle maximum action-region MSE: `3.55e-15`;
- implementation sanity: passed;
- smoke status: `diagnostic_only`, as required.

No implementation repair or setting change is justified by the smoke.

## Next action

Run the single frozen paired control exactly as recorded in `docs/pixel-control-paired-command.md` and `configs/pixel-control-paired-2026-08-21.json`. Then archive and compare its result with the fixed latent Gate 2 result.

## Boundaries

- Run the paired control once.
- Do not change seeds, thresholds, architecture, or training settings after seeing it.
- Do not rerun the latent formal experiment.
- The pixel control cannot convert latent Gate 2 into a pass.
- Do not begin Gate 3 planning.
