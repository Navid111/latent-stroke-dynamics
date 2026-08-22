# Current State

**Last updated:** 2026-08-22  
**Branch:** `main`  
**Current stage:** Stage 3 pixel-space target-guided painter  
**Status:** Foundation passed; random/exact engineering smoke ready

## Frozen completed foundation

- Gate 1 passed.
- Latent Gate 2 formally failed at 27.7% retrieval.
- Paired pixel control succeeded at 100% retrieval across all three seeds.
- Stage 3 protocol was frozen before implementation.

Do not rerun, retune, relabel, or replace these results.

## Stage 3 validation

The full local suite passed on the base M1 MacBook Air:

```text
30 passed in 2.57s
```

This validates preprocessing, deterministic candidate generation, random/exact planning, prior gates, renderer behavior, and pixel-control utilities together.

## Smoke implementation

Added `experiments/04_pixel_planner_smoke.py` with:

- one fixed 20-stroke synthetic target;
- random and exact-greedy planning;
- 20 selected strokes and 32 candidates per step;
- deterministic exact replay;
- finite-metric validation;
- progress CSVs and stroke JSON;
- final PNGs, fixed-scale error comparison, progress curves, and GIFs;
- diagnostic-only run configuration.

## Required local run

```bash
git pull
source .venv/bin/activate
python experiments/04_pixel_planner_smoke.py
```

Send the complete terminal output, `summary.csv`, `run_config.json`, `final_comparison.png`, and `progress_curves.png` for review.

## Next actions

1. Review the random/exact smoke numerically and visually.
2. Repair engineering issues only if the smoke exposes one.
3. Add a saved demonstration MLP checkpoint from train/validation data only.
4. Implement learned candidate ranking with exact execution.
5. Run a tiny all-method smoke before the controlled comparison.

## Boundaries

- The smoke is diagnostic only.
- Do not rerun completed paired experiments.
- Do not use paired test rows to select a demonstration checkpoint.
- Keep Stage 3 grayscale, 64×64, straight-line, and one-step greedy.
