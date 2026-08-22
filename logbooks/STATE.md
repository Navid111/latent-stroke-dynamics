# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Stage 3 pixel-space target-guided painter  
**Status:** Random/exact smoke passed; learned checkpoint workflow ready for local validation

## Frozen completed foundation

- Gate 1 passed.
- Latent Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval across all three seeds.
- Stage 3 protocol was frozen before implementation.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 validation

- Full suite before learned-planner code: `30 passed in 2.57s`.
- Random/exact smoke: engineering pass.
- Exact greedy reduced MSE by 50.20% after 20 strokes.
- Random reduced MSE by 3.03%.
- Exact improved all 20 steps and deterministic replay passed.
- Static images and progress curves were visually coherent.

## Learned checkpoint implementation

Added:

- `src/latent_stroke_dynamics/learned_pixel_planner.py`;
- `experiments/05_train_pixel_planner_checkpoint.py`;
- `tests/test_learned_pixel_planner.py`.

The module supports strict checkpoint metadata, state-dict hashing, safe reloading, batched candidate scores, learned one-step ranking, exact execution, and per-step exact regret diagnostics.

The demonstration trainer uses only:

- 1,000 training rows, seed `20260824`;
- 200 validation rows, seed `20260825`;
- MLP seed `11`;
- the frozen 833-parameter architecture and training settings.

It does not generate or use paired test rows.

## Required local validation

```bash
git pull
source .venv/bin/activate
pytest
```

Expected: 33 tests. If they pass, run:

```bash
python experiments/05_train_pixel_planner_checkpoint.py
```

## Next actions

1. Validate 33 tests locally.
2. Train and integrity-check the separate demonstration checkpoint.
3. Review checkpoint metadata and history.
4. Add the all-method random/exact/learned smoke.
5. Freeze and run the six-target controlled comparison.

## Boundaries

- Checkpoint weights remain local and ignored by Git.
- No paired test rows may be used for checkpoint training or selection.
- The checkpoint is a deployment artifact, not a new paired-control result.
- The learned planner always executes selected strokes with the exact renderer.
