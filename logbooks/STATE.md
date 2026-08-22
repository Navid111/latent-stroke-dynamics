# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Stage 3 pixel-space target-guided painter  
**Status:** Demonstration checkpoint validated; all-method smoke ready

## Frozen completed foundation

- Gate 1 passed.
- Latent Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval across all three seeds.
- Stage 3 protocol was frozen before implementation.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 validation

- full suite: `33 passed in 2.12s`;
- random/exact smoke: engineering pass;
- exact greedy reduced MSE by 50.20% after 20 strokes;
- random reduced MSE by 3.03%;
- deterministic exact replay passed.

## Demonstration checkpoint

The local 833-parameter MLP checkpoint completed successfully:

- path: `checkpoints/stage3-pixel-mlp-seed11.pt`;
- state digest: `e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472`;
- best epoch: 29;
- best validation balanced MSE: `0.0001646235364023596`;
- train/validation overlap: zero;
- test rows generated or used: zero;
- reload predictions identical: true;
- CPU runtime: 48.95 seconds.

The epoch-29 state was correctly selected after a small validation increase at epoch 30. The checkpoint is a deployment artifact and remains ignored by Git.

## All-method smoke

Added `experiments/06_pixel_planner_all_methods_smoke.py`. It compares random, exact, and learned planning on the same development target and budgets. The learned path records exact rank, top-1/top-5 agreement, regret, true improvement, deterministic replay, runtime, PNGs, CSVs, strokes, and GIFs.

## Required local run

```bash
git pull
source .venv/bin/activate
python experiments/06_pixel_planner_all_methods_smoke.py
```

Send complete terminal output plus summary, config, learned diagnostics, comparison, and progress figures.

## Next actions

1. Review all-method smoke numerically and visually.
2. Repair engineering issues only if exposed.
3. Freeze the exact six-target controlled command.
4. Run the controlled comparison once.
5. Add the user-image CLI and qualitative demonstrations.

## Boundaries

- Checkpoint weights remain local and ignored by Git.
- No paired test rows may be used for checkpoint training or selection.
- The all-method smoke is development-only.
- The learned planner always executes selected strokes with the exact renderer.
