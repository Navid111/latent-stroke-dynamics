# Current State

**Last updated:** 2026-08-21  
**Branch:** `main`  
**Current stage:** Stage 3 pixel-space target-guided painter  
**Status:** Scope explicitly reopened; protocol frozen before implementation

## Completed experimental foundation

- Gate 1 passed: frozen DINOv2-small patch features preserve localized strokes.
- Latent Gate 2 formally failed: 27.7% exact-action retrieval despite strong average prediction.
- Paired pixel control succeeded: 100% exact-action retrieval across all three seeds.
- Final latent and pixel figures were reviewed.
- Results and comparison drafts were started.

These results are frozen and must not be rerun or rewritten.

## Active goal

Build the requested final artifact: input an image, preprocess it to a 64×64 grayscale target, and construct an approximation line by line using sequentially selected straight strokes.

## Required planning methods

1. random candidate selection;
2. exact-renderer greedy pixel selection;
3. learned pixel-predictor selection followed by exact execution.

The learned latent predictor remains excluded because it failed exact action retrieval.

## Frozen Stage 3 settings

- six controlled 20-stroke synthetic targets;
- 100 selected strokes per painting;
- 128 candidates per step;
- error-guided plus uniform candidate proposal;
- pixel MSE target objective;
- shared deterministic budgets and seeds;
- exact execution after every selection;
- final MSE, MAE, error curves, ranking agreement, regret, runtime, PNG, JSON, and animation artifacts.

See `docs/stage-3-pixel-planner-protocol.md`.

## Next actions

1. Implement target preprocessing and deterministic candidate generation.
2. Implement random and exact-greedy planning with tests.
3. Add a saved demonstration MLP checkpoint using train/validation data only.
4. Implement learned candidate ranking.
5. Run a tiny all-method smoke before the controlled comparison.

## Boundaries

- Do not rerun or retune the completed latent or paired pixel experiments.
- Do not use paired test rows to select the demonstration checkpoint.
- Do not implement a learned latent planner.
- Keep Stage 3 grayscale, 64×64, straight-line, and one-step greedy.
- No reinforcement learning, color, textured brushes, or multi-step planning before submission.
